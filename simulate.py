import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pygame
import random
import math
from collections import deque
from tqdm import trange
import cv2
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# 공통 설정: 디바이스 & 재현성
# -----------------------------
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_writer(run_name: Optional[str] = None) -> SummaryWriter:
    run_name = run_name or f"gdash_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer

# -----------------------------
# 시각화 지오메트리 대시 환경
# -----------------------------
class VisualGeometryDashEnv(gym.Env):
    def __init__(self, render_mode=None, fps=60, record_video=False, video_path=None,
                 collect_tb_frames: bool = False, tb_video_max_frames: int = 300):
        super(VisualGeometryDashEnv, self).__init__()

        # 환경 설정
        self.width = 800
        self.height = 600
        self.ground_y = 500
        self.gravity = 0.8
        self.jump_force = -15
        self.speed = 6

        # 시각화 설정
        self.render_mode = render_mode
        self.fps = fps
        self.record_video = record_video
        self.video_path = video_path or f"geometry_dash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        # TensorBoard 비디오 프레임 수집 옵션
        self.collect_tb_frames = collect_tb_frames
        self.tb_video_max_frames = tb_video_max_frames
        self.tb_frames: List[np.ndarray] = []

        # Pygame 초기화
        if self.render_mode == "human" or self.record_video:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Geometry Dash AI")
            self.clock = pygame.time.Clock()

            # 색상 정의
            self.colors = {
                'background': (30, 30, 50),
                'ground': (100, 100, 100),
                'player': (255, 100, 100),
                'obstacle': (255, 50, 50),
                'text': (255, 255, 255)
            }

            # 비디오 녹화 설정
            if self.record_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.video_path, fourcc, self.fps, (self.width, self.height)
                )

        # 플레이어 설정
        self.player_size = 30
        self.reset_player()

        # 장애물 설정
        self.obstacles = []
        self.obstacle_spawn_distance = 200
        self.last_obstacle_x = self.width

        # Gymnasium 인터페이스
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -20, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.height, 20, self.width, self.height,
                           self.width, self.height, self.width, self.height], dtype=np.float32),
            dtype=np.float32
        )

        self.score = 0
        self.max_steps = 10000
        self.current_step = 0
        self.total_distance = 0

    def reset_player(self):
        self.player_x = 100
        self.player_y = self.ground_y - self.player_size
        self.player_vel_y = 0
        self.on_ground = True

    def spawn_obstacle(self):
        obstacle_height = random.randint(30, 80)
        obstacle = {
            'x': self.last_obstacle_x + self.obstacle_spawn_distance + random.randint(-50, 100),
            'y': self.ground_y - obstacle_height,
            'width': 20,
            'height': obstacle_height,
            'type': 'spike'
        }
        self.obstacles.append(obstacle)
        self.last_obstacle_x = obstacle['x']

    def update_obstacles(self):
        for obstacle in self.obstacles[:]:
            obstacle['x'] -= self.speed
            if obstacle['x'] + obstacle['width'] < 0:
                self.obstacles.remove(obstacle)
                self.score += 10

        if len(self.obstacles) < 5 and (not self.obstacles or self.obstacles[-1]['x'] < self.width):
            self.spawn_obstacle()

    def check_collision(self):
        # pygame.Rect 사용 (시각화 모드가 아니어도 Rect는 사용 가능)
        player_rect = pygame.Rect(self.player_x, self.player_y, self.player_size, self.player_size)
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(obstacle['x'], obstacle['y'], obstacle['width'], obstacle['height'])
            if player_rect.colliderect(obstacle_rect):
                return True
        return False

    def get_state(self):
        state = [
            self.player_y / self.height,
            self.player_vel_y / 20.0,
        ]

        upcoming_obstacles = [obs for obs in self.obstacles if obs['x'] > self.player_x][:3]
        for i in range(3):
            if i < len(upcoming_obstacles):
                obs = upcoming_obstacles[i]
                state.extend([
                    (obs['x'] - self.player_x) / self.width,
                    obs['y'] / self.height
                ])
            else:
                state.extend([1.0, 0.5])

        return np.array(state, dtype=np.float32)

    def _grab_rgb_frame(self) -> np.ndarray:
        # RGB(H, W, 3), dtype=uint8
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))  # (W,H,3) -> (H,W,3)
        return frame

    def render(self):
        if self.render_mode != "human" and not self.record_video and not self.collect_tb_frames:
            return

        # 배경
        self.screen.fill(self.colors['background'])

        # 지면
        pygame.draw.rect(self.screen, self.colors['ground'],
                         (0, self.ground_y, self.width, self.height - self.ground_y))

        # 플레이어 (회전)
        rotation_angle = (self.current_step * 5) % 360 if not self.on_ground else 0
        player_surface = pygame.Surface((self.player_size, self.player_size), pygame.SRCALPHA)
        pygame.draw.rect(player_surface, self.colors['player'], (0, 0, self.player_size, self.player_size))
        rotated_player = pygame.transform.rotate(player_surface, rotation_angle)
        player_rect = rotated_player.get_rect(center=(self.player_x + self.player_size//2,
                                                     self.player_y + self.player_size//2))
        self.screen.blit(rotated_player, player_rect)

        # 장애물
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.colors['obstacle'],
                             (obstacle['x'], obstacle['y'], obstacle['width'], obstacle['height']))
            points = [
                (obstacle['x'], obstacle['y'] + obstacle['height']),
                (obstacle['x'] + obstacle['width']//2, obstacle['y']),
                (obstacle['x'] + obstacle['width'], obstacle['y'] + obstacle['height'])
            ]
            pygame.draw.polygon(self.screen, (255, 0, 0), points)

        # UI
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.total_distance//10}", True, self.colors['text'])
        step_text = font.render(f"Steps: {self.current_step}", True, self.colors['text'])
        velocity_text = font.render(f"Velocity: {self.player_vel_y:.1f}", True, self.colors['text'])
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(step_text, (10, 50))
        self.screen.blit(velocity_text, (10, 90))

        if self.obstacles:
            next_obstacle = min(self.obstacles, key=lambda obs: abs(obs['x'] - self.player_x))
            distance = next_obstacle['x'] - self.player_x
            distance_text = font.render(f"Next: {distance:.0f}px", True, self.colors['text'])
            self.screen.blit(distance_text, (10, 130))

        pygame.display.flip()

        # 프레임 캡처
        if self.record_video or self.collect_tb_frames:
            frame_rgb = self._grab_rgb_frame()

        # 비디오 파일 저장용(BGR)
        if self.record_video:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)

        # TensorBoard 비디오용(RGB)
        if self.collect_tb_frames and len(self.tb_frames) < self.tb_video_max_frames:
            self.tb_frames.append(frame_rgb)

        if self.render_mode == "human":
            self.clock.tick(self.fps)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.reset_player()
        self.obstacles = []
        self.last_obstacle_x = self.width
        self.score = 0
        self.current_step = 0
        self.total_distance = 0
        self.tb_frames = []

        for _ in range(3):
            self.spawn_obstacle()

        if self.render_mode == "human" or self.record_video or self.collect_tb_frames:
            self.render()

        return self.get_state(), {}

    def step(self, action):
        self.current_step += 1
        self.total_distance += self.speed

        # 이벤트 처리
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return self.get_state(), 0.0, True, False, {}

        # 액션 처리
        if action == 1 and self.on_ground:
            self.player_vel_y = self.jump_force
            self.on_ground = False

        # 물리 업데이트
        self.player_vel_y += self.gravity
        self.player_y += self.player_vel_y

        # 지면 충돌
        if self.player_y >= self.ground_y - self.player_size:
            self.player_y = self.ground_y - self.player_size
            self.player_vel_y = 0
            self.on_ground = True

        # 장애물 업데이트
        self.update_obstacles()

        # 보상
        reward = 0.1
        terminated = False

        if self.check_collision():
            reward = -100.0
            terminated = True
        elif self.current_step >= self.max_steps:
            terminated = True
            reward += 50.0

        reward += self.score * 0.1
        self.score = 0

        # 렌더링
        if self.render_mode == "human" or self.record_video or self.collect_tb_frames:
            self.render()

        return self.get_state(), float(reward), bool(terminated), False, {}

    def get_tb_video_tensor(self) -> Optional[torch.Tensor]:
        """수집된 프레임을 TensorBoard 비디오 텐서(N,T,C,H,W, [0,1])로 변환"""
        if not self.tb_frames:
            return None
        frames = np.stack(self.tb_frames, axis=0)  # (T,H,W,3), uint8
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)  # (1,T,3,H,W)
        frames = frames.to(torch.float32) / 255.0
        return frames

    def close(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        if hasattr(self, 'screen'):
            pygame.quit()

# PPO 네트워크
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

# PPO 에이전트 (TensorBoard 메트릭 계산용 보강)
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, entropy_coef=0.01):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef

        self.network = PPONetwork(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu', enabled=AMP_ENABLED)

        self.memory = []

    @torch.no_grad()
    def select_action(self, state_np):
        state = torch.as_tensor(state_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action_probs, _ = self.network(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), float(logp.item())

    def store_experience(self, state, action, reward, log_prob, done):
        self.memory.append((state, action, reward, log_prob, done))

    def _calc_explained_variance(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        # y_true: returns, y_pred: values
        var_y = torch.var(y_true)
        if var_y.item() < 1e-8:
            return 0.0
        return float(1.0 - torch.var(y_true - y_pred) / var_y)

    def update(self) -> Dict[str, float]:
        """PPO 업데이트 한 번 수행하고, 로깅용 메트릭 반환"""
        metrics: Dict[str, float] = {}
        if len(self.memory) == 0:
            return metrics

        # 메모리 텐서화
        states_np = np.array([exp[0] for exp in self.memory])
        states = torch.as_tensor(states_np, dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor([exp[1] for exp in self.memory], dtype=torch.long, device=DEVICE)
        rewards = [exp[2] for exp in self.memory]
        dones = [exp[4] for exp in self.memory]
        old_log_probs = torch.as_tensor([exp[3] for exp in self.memory], dtype=torch.float32, device=DEVICE)

        # 할인 리턴
        discounted_rewards = []
        discounted = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                discounted = 0.0
            discounted = float(r) + self.gamma * discounted
            discounted_rewards.insert(0, discounted)
        returns = torch.as_tensor(discounted_rewards, dtype=torch.float32, device=DEVICE)
        # 표준화
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        # 누적 메트릭
        actor_losses, critic_losses, entropies, kls, clip_fracs, ratios_means = [], [], [], [], [], []
        grad_pre_norms, grad_post_norms = [], []

        for _ in range(self.k_epochs):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP_ENABLED):
                action_probs, values = self.network(states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # ratio & clipped obj
                ratio = torch.exp(new_log_probs - old_log_probs)
                values = values.squeeze(-1)
                advantages = (returns - values).detach()

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, returns)
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                # 추가 메트릭
                approx_kl = (old_log_probs - new_log_probs).mean().abs()
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.eps_clip).float())

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()

            # 그래드 노름(클리핑 전/후)
            self.scaler.unscale_(self.optimizer)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            grad_pre_norms.append(float(pre_clip_norm))
            # post norm은 이론상 <= max_norm
            grad_post_norms.append(float(min(pre_clip_norm.item(), 0.5)))

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # epoch 메트릭 수집
            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            kls.append(float(approx_kl.detach().cpu()))
            clip_fracs.append(float(clip_frac.detach().cpu()))
            ratios_means.append(float(ratio.mean().detach().cpu()))

        # 에폭 평균 집계
        metrics.update({
            "loss/actor": float(np.mean(actor_losses)),
            "loss/critic": float(np.mean(critic_losses)),
            "policy/entropy": float(np.mean(entropies)),
            "policy/kl": float(np.mean(kls)),
            "policy/clip_fraction": float(np.mean(clip_fracs)),
            "policy/ratio_mean": float(np.mean(ratios_means)),
            "grad/pre_clip_norm": float(np.mean(grad_pre_norms)),
            "grad/post_clip_norm": float(np.mean(grad_post_norms)),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        })

        # value/adv 통계 & explained variance
        with torch.no_grad():
            action_probs, values = self.network(states)
            values = values.squeeze(-1)
            adv = returns - values
            ev = self._calc_explained_variance(returns, values)
            metrics.update({
                "value/explained_variance": ev,
                "value/values_mean": float(values.mean().cpu()),
                "value/values_std": float(values.std(unbiased=False).cpu()),
                "adv/mean": float(adv.mean().cpu()),
                "adv/std": float(adv.std(unbiased=False).cpu()),
            })
            # 정책 분포 히스토그램용 일부 표본
            metrics["policy/prob_mean"] = float(action_probs.mean().cpu())

        self.memory.clear()
        return metrics

# -----------------------------
# 시각화 + TensorBoard 훈련
# -----------------------------
def train_with_visualization(
    episodes=1000,
    max_timesteps=2000,
    update_timestep=2000,
    render_mode=None,              # "human" or None
    record_video=False,
    video_episodes=(0, 100, 200, 500, 800),  # 파일로 녹화
    tb_video_episodes=(0, 200, 800),         # TensorBoard 비디오 기록
    tb_video_max_frames=300,
    seed=42,
    run_name: Optional[str] = None
):
    set_seed(seed)

    writer = create_writer(run_name)
    # 하이퍼파라미터 기록
    hparams = {
        "episodes": episodes,
        "max_timesteps": max_timesteps,
        "update_timestep": update_timestep,
        "device": DEVICE.type,
        "amp": AMP_ENABLED,
        "lr": 3e-4,
        "gamma": 0.99,
        "eps_clip": 0.2,
        "k_epochs": 4,
        "entropy_coef": 0.01,
    }
    writer.add_text("run/info", f"Device: {DEVICE.type}, AMP: {AMP_ENABLED}")
    writer.add_text("run/command", "tensorboard --logdir runs")

    timestep = 0
    update_count = 0
    episode_rewards = deque(maxlen=100)

    print(f"[Device] {DEVICE.type.upper()} | AMP: {AMP_ENABLED}")
    print("시각화 지오메트리 대시 AI 훈련 시작!")

    # 첫 에피소드에서 에이전트 생성
    agent = PPOAgent(state_dim=8, action_dim=2)

    # 그래프 기록(선택)
    try:
        dummy = torch.zeros(1, 8, dtype=torch.float32).to(DEVICE)
        writer.add_graph(agent.network, dummy)
    except Exception:
        pass  # 환경에 따라 graph 추적이 실패할 수 있음

    ckpt_template = "geometry_dash_ppo_episode_{ep}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    try:
        pbar = trange(episodes, desc="Episodes", ncols=100)
        for episode in pbar:
            # 렌더링/비디오/텐서보드 비디오 조건
            should_show = (render_mode == "human") and (episode % 50 == 0)
            should_record_file = record_video and (episode in set(video_episodes))
            should_record_tb = (episode in set(tb_video_episodes))

            env = VisualGeometryDashEnv(
                render_mode="human" if should_show else None,
                record_video=should_record_file,
                video_path=f"geometry_dash_episode_{episode}.mp4" if should_record_file else None,
                collect_tb_frames=should_record_tb,
                tb_video_max_frames=tb_video_max_frames
            )

            state, _ = env.reset()
            episode_reward = 0.0
            ep_len = 0

            for t in range(max_timesteps):
                action, log_prob = agent.select_action(state)
                next_state, reward, terminated, _, _ = env.step(action)

                agent.store_experience(state, action, reward, log_prob, terminated)

                state = next_state
                episode_reward += float(reward)
                timestep += 1
                ep_len += 1

                if timestep % update_timestep == 0:
                    metrics = agent.update()
                    update_count += 1
                    # --- TensorBoard 로깅(업데이트 스텝 기준) ---
                    if metrics:
                        for k, v in metrics.items():
                            writer.add_scalar(k, v, global_step=update_count)
                        # 가끔 파라미터 히스토그램 기록
                        if update_count % 200 == 0:
                            for name, p in agent.network.named_parameters():
                                if p.grad is not None:
                                    writer.add_histogram(f"params/{name}", p.detach().cpu(), global_step=update_count)
                                    writer.add_histogram(f"grads/{name}", p.grad.detach().cpu(), global_step=update_count)

                if terminated:
                    break

            env.close()
            episode_rewards.append(episode_reward)

            # --- TensorBoard 로깅(에피소드 기준) ---
            writer.add_scalar("train/episode_reward", episode_reward, global_step=episode)
            writer.add_scalar("train/episode_length", ep_len, global_step=episode)
            if len(episode_rewards) > 0:
                avg100 = float(np.mean(episode_rewards))
                writer.add_scalar("train/avg_reward_100", avg100, global_step=episode)

            # TQDM 표시
            if episode % 50 == 0:
                avg_reward = float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else episode_reward
                pbar.set_postfix(avg_reward=f"{avg_reward:.2f}", last_ep=f"{episode_reward:.2f}")

            # TensorBoard 비디오 기록
            if should_record_tb:
                video_tensor = env.get_tb_video_tensor()
                if video_tensor is not None:
                    writer.add_video("video/episode", video_tensor, global_step=episode, fps=env.fps)

            # 체크포인트
            if (episode % 200) == 0:
                ckpt_path = os.path.join("checkpoints", ckpt_template.format(ep=episode))
                torch.save(agent.network.state_dict(), ckpt_path)
                writer.add_text("checkpoint/saved", ckpt_path, global_step=episode)

        print("훈련 완료!")

    except KeyboardInterrupt:
        print("훈련 중단됨")
    finally:
        # 마지막 체크포인트 + hparams 요약
        final_ckpt = os.path.join("checkpoints", ckpt_template.format(ep="final"))
        torch.save(agent.network.state_dict(), final_ckpt)
        writer.add_text("checkpoint/final", final_ckpt)

        # hparams 결과(최종 평균 보상)
        final_avg = float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else 0.0
        writer.add_hparams(hparams, {"metric/avg_reward_100": final_avg})
        writer.close()

    return agent

# -----------------------------
# 테스트 (시각화 + 선택적 비디오)
# -----------------------------
@torch.no_grad()
def test_with_visualization(model_path, episodes=3, render_mode="human", record_video=True):
    env = VisualGeometryDashEnv(
        render_mode=render_mode,
        record_video=record_video,
        video_path="geometry_dash_test.mp4"
    )

    agent = PPOAgent(state_dim=8, action_dim=2)
    agent.network.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.network.eval()

    for i in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action_probs, _ = agent.network(s)
            action = int(torch.argmax(action_probs, dim=-1).item())

            state, reward, terminated, _, _ = env.step(action)
            total_reward += float(reward)
            steps += 1

            if terminated:
                break

        print(f"[Episode {i}] 총 보상: {total_reward:.2f}, 생존 스텝: {steps}")

    env.close()

# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    trained_agent = train_with_visualization(
        episodes=1000,
        max_timesteps=2000,
        update_timestep=2000,
        render_mode=None,            # "human"으로 두면 50에피소드마다 렌더링
        record_video=True,
        video_episodes=(0, 100, 200, 500, 800),
        tb_video_episodes=(0, 200, 800),
        tb_video_max_frames=300,
        seed=42,
        run_name=None                # None이면 자동 타임스탬프 이름
    )

    # 필요 시 테스트
    # test_with_visualization('checkpoints/geometry_dash_ppo_episode_final.pth', episodes=3)

