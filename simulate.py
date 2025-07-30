# -*- coding: utf-8 -*-
"""
Geometry Dash PPO+RND (SOTA RL stack)
- PPO: GAE(λ), mini-batch, value clipping, adaptive KL early stop
- Schedules: LR warmup+cosine decay, entropy linear decay
- Exploration: RND intrinsic reward
- VecEnv + StateStack + RunningMeanStd
- CUDA + AMP, gradient clipping
- TQDM + TensorBoard logging, checkpoints(best/final)
- Headless training, optional visualization for test
"""

import os
os.environ["SDL_AUDIODRIVER"] = "dummy"          # 오디오 비활성(헤드리스)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import math
import time
import random
import argparse
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Any, List

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# 전역: 디바이스/AMP/시드
# -----------------------------
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = (DEVICE.type == "cuda")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

# =========================================================
# 1) 학습용 Headless 환경 (커리큘럼 + 도메인 랜덤화)
# =========================================================
class HeadlessGeometryDashEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None, curriculum=False, domain_rand=True, difficulty=1.0):
        super().__init__()
        self.width, self.height = 800, 600
        self.base_ground_y = 500
        self.base_gravity = 0.8
        self.base_jump = -15
        self.base_speed = 6

        self.curriculum = curriculum
        self.domain_rand = domain_rand
        self.difficulty = difficulty

        self.player_size = 30
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -20, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.height, 20, self.width, self.height,
                           self.width, self.height, self.width, self.height], dtype=np.float32),
            dtype=np.float32
        )
        self._rng = np.random.RandomState(seed)
        self.reset()

    # ---- 내부 로직 ----
    def _apply_difficulty(self):
        d = float(self.difficulty)
        self.ground_y = self.base_ground_y
        self.speed   = self.base_speed * (1.0 + 0.20*(d-1.0))
        self.gravity = self.base_gravity * (1.0 + 0.10*(d-1.0))
        self.jump_force = self.base_jump * (1.0 + 0.05*(d-1.0))
        if self.domain_rand:
            self.gravity    *= self._rng.uniform(0.95, 1.05)
            self.jump_force *= self._rng.uniform(0.97, 1.03)
            self.speed      *= self._rng.uniform(0.95, 1.05)

    def _reset_player(self):
        self.player_x = 100
        self.player_y = self.ground_y - self.player_size
        self.player_vel_y = 0.0
        self.on_ground = True

    def _spawn_obstacle(self):
        h = self._rng.randint(30, int(80 * (0.9 + 0.2*self.difficulty)))
        gap = 200 + self._rng.randint(-50, 100)
        obs = {'x': self.last_obstacle_x + gap, 'y': self.ground_y - h, 'width': 20, 'height': h}
        self.obstacles.append(obs)
        self.last_obstacle_x = obs['x']

    def _update_obstacles(self):
        for obs in self.obstacles[:]:
            obs['x'] -= self.speed
            if obs['x'] + obs['width'] < 0:
                self.obstacles.remove(obs)
                self.score += 10
        if len(self.obstacles) < 5 and (not self.obstacles or self.obstacles[-1]['x'] < self.width):
            self._spawn_obstacle()

    def _collide(self):
        px, py, ps = self.player_x, self.player_y, self.player_size
        for o in self.obstacles:
            if (px < o['x'] + o['width'] and px + ps > o['x'] and
                py < o['y'] + o['height'] and py + ps > o['y']):
                return True
        return False

    def _state(self):
        s = [self.player_y / self.height, self.player_vel_y / 20.0]
        ups = [o for o in self.obstacles if o['x'] > self.player_x][:3]
        for i in range(3):
            if i < len(ups):
                o = ups[i]
                s += [(o['x'] - self.player_x) / self.width, o['y'] / self.height]
            else:
                s += [1.0, 0.5]
        return np.array(s, dtype=np.float32)

    # ---- Gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._apply_difficulty()
        self._reset_player()
        self.obstacles = []
        self.last_obstacle_x = self.width
        self.score = 0.0
        self.step_count = 0
        self.max_steps = 10_000
        for _ in range(3):
            self._spawn_obstacle()
        return self._state(), {}

    def step(self, action):
        self.step_count += 1
        if action == 1 and self.on_ground:
            self.player_vel_y = self.jump_force
            self.on_ground = False

        self.player_vel_y += self.gravity
        self.player_y += self.player_vel_y
        if self.player_y >= self.ground_y - self.player_size:
            self.player_y = self.ground_y - self.player_size
            self.player_vel_y = 0.0
            self.on_ground = True

        self._update_obstacles()

        reward = 0.1 + self.score * 0.1
        self.score = 0.0

        terminated = False
        if self._collide():
            reward = -100.0
            terminated = True
        elif self.step_count >= self.max_steps:
            reward += 50.0
            terminated = True

        return self._state(), float(reward), terminated, False, {}

# =========================================================
# 2) 래퍼/정규화/탐색(RND)
# =========================================================
class StateStackWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        low  = np.repeat(env.observation_space.low,  k, axis=0)
        high = np.repeat(env.observation_space.high, k, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.frames = deque(maxlen=k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._obs(), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._obs(), r, term, trunc, info

    def _obs(self):
        return np.concatenate(list(self.frames), axis=0)

class RunningMeanStd:
    def __init__(self, eps=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var  = np.ones(shape, 'float64')
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        var  = x.var(axis=0)
        n    = x.shape[0]
        delta = mean - self.mean
        tot = self.count + n
        new_mean = self.mean + delta * n / tot
        m_a = self.var * self.count
        m_b = var * n
        M2 = m_a + m_b + delta**2 * self.count * n / tot
        new_var = M2 / tot
        self.mean, self.var, self.count = new_mean, new_var, tot

# ---- RND ----
class RNDNet(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 64)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class RNDModule:
    def __init__(self, obs_dim, lr=1e-4, device=DEVICE, obs_clip=5.0, bonus_scale=0.05):
        self.device = device
        self.target = RNDNet(obs_dim).to(device)
        self.predictor = RNDNet(obs_dim).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_clip = obs_clip
        self.bonus_scale = bonus_scale
        self.criterion = nn.MSELoss(reduction='none')

    @torch.no_grad()
    def compute_bonus(self, obs_t: torch.Tensor) -> torch.Tensor:
        o = torch.clamp(obs_t, -self.obs_clip, self.obs_clip)
        t = self.target(o)
        p = self.predictor(o)
        err = (t - p).pow(2).mean(dim=-1)  # (B,)
        norm = (err - err.mean()) / (err.std(unbiased=False) + 1e-8)
        return self.bonus_scale * norm

    def update(self, obs_t: torch.Tensor) -> float:
        o = torch.clamp(obs_t, -self.obs_clip, self.obs_clip)
        with torch.no_grad():
            tgt = self.target(o)
        pred = self.predictor(o)
        loss = self.criterion(pred, tgt).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach().cpu().item())

# =========================================================
# 3) PPO 네트워크/버퍼/트레이너 (SOTA 안정화)
# =========================================================
def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)

class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
        self.apply(orthogonal_init)

    def forward(self, x):
        h = self.feature(x)
        return self.actor(h), self.critic(h)

class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_dim, device=DEVICE):
        self.n_steps, self.n_envs, self.device = n_steps, n_envs, device
        self.obs = torch.zeros(n_steps+1, n_envs, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, device=device)
        self.logps = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps+1, n_envs, device=device)
        self.ptr = 0

    def add(self, obs, action, reward, done, logp, value):
        t = self.ptr
        self.obs[t].copy_(obs)
        self.actions[t].copy_(action)
        self.rewards[t].copy_(reward)
        self.dones[t].copy_(done)
        self.logps[t].copy_(logp)
        self.values[t].copy_(value)
        self.ptr += 1

    def finish(self, last_value, gamma, lam):
        self.values[self.ptr].copy_(last_value)
        T = self.ptr
        adv = torch.zeros(T, self.n_envs, device=self.device)
        last_gae = torch.zeros(self.n_envs, device=self.device)
        for t in reversed(range(T)):
            nonterm = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * self.values[t+1] * nonterm - self.values[t]
            last_gae = delta + gamma * lam * nonterm * last_gae
            adv[t] = last_gae
        ret = adv + self.values[:T]
        data = {
            "obs": self.obs[:T].reshape(-1, self.obs.size(-1)),
            "actions": self.actions[:T].reshape(-1),
            "logps": self.logps[:T].reshape(-1),
            "advantages": (adv.reshape(-1) - adv.mean())/(adv.std(unbiased=False)+1e-8),
            "returns": ret.reshape(-1),
            "old_values": self.values[:T].reshape(-1),
        }
        self.ptr = 0
        return data

class PPOTrainer:
    def __init__(self, net, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4,
                 vf_coef=0.5, ent_coef=0.01, target_kl=0.03, max_grad_norm=0.5, minibatch_size=2048):
        self.net = net
        self.gamma, self.lam = gamma, lam
        self.clip, self.epochs = clip, epochs
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size

        self.opt = optim.Adam(net.parameters(), lr=lr, eps=1e-5)
        # LR 스케줄러: warmup -> cosine decay
        def lr_lambda(step):
            warmup = 50
            if step < warmup: 
                return (step + 1) / warmup
            return 0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * (step - warmup) / 2000)))
        self.sched = optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)

        # 엔트로피 선형 감소
        self.ent_start, self.ent_end = ent_coef, max(0.001, ent_coef * 0.1)
        self.update_step = 0
        self.scaler = torch.amp.GradScaler('cuda' if AMP else 'cpu', enabled=AMP)

    def evaluate(self, obs):
        pi, v = self.net(obs)
        return Categorical(pi), v.squeeze(-1)

    def ent_coef_now(self):
        T = 2000  # 총 업데이트 가정
        r = min(1.0, self.update_step / T)
        return self.ent_start * (1.0 - r) + self.ent_end * r

    def update(self, batch, writer: Optional[SummaryWriter] = None, gs: int = 0) -> Dict[str, float]:
        obs = batch["obs"]; actions = batch["actions"]
        old_logps = batch["logps"]; adv = batch["advantages"]
        returns = batch["returns"]; old_values = batch["old_values"]

        n = obs.size(0)
        idx = torch.randperm(n, device=obs.device)
        agg = {"loss_pi":0.0, "loss_v":0.0, "ent":0.0, "kl":0.0, "clip_frac":0.0}

        stop_early = False
        for _ in range(self.epochs):
            for s in range(0, n, self.minibatch_size):
                e = min(s + self.minibatch_size, n)
                mb = idx[s:e]
                mb_obs, mb_act = obs[mb], actions[mb]
                mb_old_logp, mb_adv = old_logps[mb], adv[mb]
                mb_ret, mb_old_v = returns[mb], old_values[mb]

                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
                    dist, v = self.evaluate(mb_obs)
                    new_logp = dist.log_prob(mb_act)
                    ratio = torch.exp(new_logp - mb_old_logp)

                    # policy loss with clipping
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * mb_adv
                    pi_loss = -torch.min(surr1, surr2).mean()

                    # value loss with clipping
                    v_clip = mb_old_v + torch.clamp(v - mb_old_v, -self.clip, self.clip)
                    v_loss = 0.5 * torch.max((v - mb_ret).pow(2), (v_clip - mb_ret).pow(2)).mean()

                    ent = dist.entropy().mean()
                    ent_coef = self.ent_coef_now()
                    loss = pi_loss + self.vf_coef * v_loss - ent_coef * ent

                    approx_kl = (mb_old_logp - new_logp).mean().abs()
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip).float().mean()

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()

                agg["loss_pi"] += float(pi_loss.detach().cpu())
                agg["loss_v"]  += float(v_loss.detach().cpu())
                agg["ent"]     += float(ent.detach().cpu())
                agg["kl"]      += float(approx_kl.detach().cpu())
                agg["clip_frac"] += float(clip_frac.detach().cpu())

                if approx_kl > self.target_kl:
                    stop_early = True
                    break
            if stop_early: 
                break

        self.sched.step()
        self.update_step += 1

        denom = math.ceil(n/self.minibatch_size) * (1 if stop_early else self.epochs)
        for k in agg: agg[k] /= max(1, denom)

        if writer is not None:
            writer.add_scalar("loss/actor", agg["loss_pi"], gs)
            writer.add_scalar("loss/critic", agg["loss_v"], gs)
            writer.add_scalar("policy/entropy", agg["ent"], gs)
            writer.add_scalar("policy/kl", agg["kl"], gs)
            writer.add_scalar("policy/clip_fraction", agg["clip_frac"], gs)
            writer.add_scalar("opt/lr", self.opt.param_groups[0]["lr"], gs)
            writer.add_scalar("opt/ent_coef", self.ent_coef_now(), gs)

        return agg

# =========================================================
# 4) 학습 루프 (VecEnv + RND + TB 로깅 + 체크포인트)
# =========================================================
def make_env(seed_base=0, curriculum=True, domain_rand=True):
    def _thunk():
        env = HeadlessGeometryDashEnv(seed=seed_base, curriculum=curriculum, domain_rand=domain_rand)
        env = StateStackWrapper(env, k=4)
        return env
    return _thunk

def train_ppo_rnd(
    total_updates=2000,
    n_envs=16,
    n_steps=512,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    clip=0.2,
    epochs=4,
    minibatch_size=2048,
    ent_coef=0.01,
    target_kl=0.03,
    rnd_scale=0.05,
    run_name: Optional[str]=None,
    seed: int = 42,
    curriculum: bool = True
):
    set_seed(seed)
    os.makedirs("checkpoints", exist_ok=True)
    run_name = run_name or f"gdash_sota_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    print(f"[Device] {DEVICE.type.upper()} | AMP: {AMP}")
    writer.add_text("run/info", f"Device: {DEVICE.type}, AMP: {AMP}")
    hparams = {
        "total_updates": total_updates, "n_envs": n_envs, "n_steps": n_steps, "batch": n_envs*n_steps,
        "lr": lr, "gamma": gamma, "lam": lam, "clip": clip, "epochs": epochs,
        "minibatch_size": minibatch_size, "ent_coef": ent_coef, "target_kl": target_kl,
        "rnd_scale": rnd_scale, "seed": seed, "curriculum": curriculum
    }

    # VecEnv
    env_fns = [make_env(seed_base=seed+i, curriculum=curriculum, domain_rand=True) for i in range(n_envs)]
    vec_env = gym.vector.SyncVectorEnv(env_fns)

    # 초기화
    obs, _ = vec_env.reset()
    obs_dim = obs.shape[1]
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    net = PPONetwork(state_dim=obs_dim, action_dim=2).to(DEVICE)
    trainer = PPOTrainer(net, lr=lr, gamma=gamma, lam=lam, clip=clip, epochs=epochs,
                         vf_coef=0.5, ent_coef=ent_coef, target_kl=target_kl,
                         max_grad_norm=0.5, minibatch_size=minibatch_size)
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim, DEVICE)

    rnd = RNDModule(obs_dim=obs_dim, lr=1e-4, device=DEVICE, bonus_scale=rnd_scale)

    global_step = 0
    best_ret = -1e9

    # (선택) 네트워크 그래프 기록
    try:
        dummy = torch.zeros(1, obs_dim, dtype=torch.float32, device=DEVICE)
        writer.add_graph(net, dummy)
    except Exception:
        pass

    pbar = trange(total_updates, desc="Updates", ncols=100)
    for upd in pbar:
        ep_ext_returns = []
        ep_int_returns = []
        rnd_losses = []

        for t in range(n_steps):
            # 관측 정규화
            obs_rms.update(obs)
            obs_norm = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
            obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                pi, v = net(obs_t)
                dist = Categorical(pi)
                actions = dist.sample()
                logps = dist.log_prob(actions)

            next_obs, rewards, terms, truncs, infos = vec_env.step(actions.cpu().numpy())
            dones = np.logical_or(terms, truncs).astype(np.float32)

            with torch.no_grad():
                bonus = rnd.compute_bonus(obs_t)  # (B,)
            total_reward = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE) + bonus

            # RND 예측기 업데이트(모든 스텝) — 필요시 t%2==0 등으로 줄일 수 있음
            rnd_loss = rnd.update(obs_t)
            rnd_losses.append(rnd_loss)

            buffer.add(
                obs=obs_t,
                action=actions,
                reward=total_reward,
                done=torch.as_tensor(dones, dtype=torch.float32, device=DEVICE),
                logp=logps.detach(),
                value=v.squeeze(-1).detach()
            )

            obs = next_obs
            global_step += n_envs
            ep_ext_returns.append(float(np.mean(rewards)))
            ep_int_returns.append(float(bonus.mean().detach().cpu()))

        # 부트스트랩 값
        obs_norm = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            _, last_v = net(obs_t)

        batch = buffer.finish(last_v.squeeze(-1), gamma, lam)
        metrics = trainer.update(batch, writer=writer, gs=upd)

        mean_ext = np.mean(ep_ext_returns) if ep_ext_returns else 0.0
        mean_int = np.mean(ep_int_returns) if ep_int_returns else 0.0
        mean_ret_batch = float(batch["returns"].mean().cpu())
        writer.add_scalar("reward/extrinsic_mean", mean_ext, upd)
        writer.add_scalar("reward/intrinsic_mean", mean_int, upd)
        writer.add_scalar("reward/rnd_loss", np.mean(rnd_losses) if rnd_losses else 0.0, upd)
        writer.add_scalar("train/mean_return_batch", mean_ret_batch, upd)

        pbar.set_postfix(mean_ret=f"{mean_ret_batch:.2f}", ext=f"{mean_ext:.2f}", intr=f"{mean_int:.3f}")

        # 베스트 체크포인트
        if mean_ret_batch > best_ret:
            best_ret = mean_ret_batch
            path = f"checkpoints/ppo_rnd_best.pt"
            torch.save(net.state_dict(), path)
            writer.add_text("checkpoint/best", path, upd)

        # 커리큘럼(가이드): 조건 충족 시 난이도 상향 권고 로그
        if curriculum and (upd+1) % 100 == 0 and mean_ret_batch > -20:
            writer.add_text("curriculum/info", f"Consider increasing difficulty at update {upd+1}", upd)

    # 종료 처리
    final_ckpt = "checkpoints/ppo_rnd_final.pt"
    torch.save(net.state_dict(), final_ckpt)
    writer.add_text("checkpoint/final", final_ckpt)
    writer.add_hparams(hparams, {"metric/best_mean_return": best_ret})
    writer.close()
    vec_env.close()
    print("훈련 완료! 최종 체크포인트:", final_ckpt)
    return final_ckpt

# =========================================================
# 5) (옵션) 시각화 테스트
# =========================================================
def test_visual(model_path: str, episodes: int = 3):
    """간단 테스트(헤드리스로 step만 진행; pygame 시각화는 생략)"""
    env = HeadlessGeometryDashEnv(seed=123, curriculum=False, domain_rand=False, difficulty=1.0)
    env = StateStackWrapper(env, k=4)

    net = PPONetwork(state_dim=env.observation_space.shape[0], action_dim=2).to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()

    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done:
            obs_norm = obs  # 여기선 정규화를 생략(간단 평가용). 필요시 RMS를 저장/로드하여 동일 적용.
            s = torch.as_tensor(obs_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                pi, _ = net(s)
                act = int(torch.argmax(pi, dim=-1).item())
            obs, r, term, trunc, _ = env.step(act)
            total_r += float(r)
            steps += 1
            done = term or trunc
        print(f"[Episode {i}] Return={total_r:.2f}, Steps={steps}")

# =========================================================
# 6) 엔트리포인트
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Geometry Dash PPO+RND (SOTA)")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--mb_size", type=int, default=2048)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--target_kl", type=float, default=0.03)
    p.add_argument("--rnd_scale", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_curriculum", action="store_true")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--test_after", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    final_ckpt = train_ppo_rnd(
        total_updates=args.updates,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip=args.clip,
        epochs=args.epochs,
        minibatch_size=args.mb_size,
        ent_coef=args.ent,
        target_kl=args.target_kl,
        rnd_scale=args.rnd_scale,
        run_name=args.run_name,
        seed=args.seed,
        curriculum=(not args.no_curriculum)
    )
    if args.test_after:
        test_visual(final_ckpt, episodes=3)

