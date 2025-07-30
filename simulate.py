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

# -----------------------------
# 시각화 지오메트리 대시 환경
# -----------------------------
class VisualGeometryDashEnv(gym.Env):
    def __init__(self, render_mode=None, fps=60, record_video=False, video_path=None):
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

    def render(self):
        if self.render_mode != "human" and not self.record_video:
            return
            
        # 배경 그리기
        self.screen.fill(self.colors['background'])
        
        # 지면 그리기
        pygame.draw.rect(self.screen, self.colors['ground'], 
                        (0, self.ground_y, self.width, self.height - self.ground_y))
        
        # 플레이어 그리기 (회전 효과)
        rotation_angle = (self.current_step * 5) % 360 if not self.on_ground else 0
        player_surface = pygame.Surface((self.player_size, self.player_size), pygame.SRCALPHA)
        pygame.draw.rect(player_surface, self.colors['player'], (0, 0, self.player_size, self.player_size))
        rotated_player = pygame.transform.rotate(player_surface, rotation_angle)
        player_rect = rotated_player.get_rect(center=(self.player_x + self.player_size//2, 
                                                     self.player_y + self.player_size//2))
        self.screen.blit(rotated_player, player_rect)
        
        # 장애물 그리기
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.colors['obstacle'],
                           (obstacle['x'], obstacle['y'], obstacle['width'], obstacle['height']))
            # 스파이크 효과
            points = [
                (obstacle['x'], obstacle['y'] + obstacle['height']),
                (obstacle['x'] + obstacle['width']//2, obstacle['y']),
                (obstacle['x'] + obstacle['width'], obstacle['y'] + obstacle['height'])
            ]
            pygame.draw.polygon(self.screen, (255, 0, 0), points)
        
        # UI 정보 표시
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.total_distance//10}", True, self.colors['text'])
        step_text = font.render(f"Steps: {self.current_step}", True, self.colors['text'])
        velocity_text = font.render(f"Velocity: {self.player_vel_y:.1f}", True, self.colors['text'])
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(step_text, (10, 50))
        self.screen.blit(velocity_text, (10, 90))
        
        # 다음 장애물까지 거리 표시
        if self.obstacles:
            next_obstacle = min(self.obstacles, key=lambda obs: abs(obs['x'] - self.player_x))
            distance = next_obstacle['x'] - self.player_x
            distance_text = font.render(f"Next: {distance:.0f}px", True, self.colors['text'])
            self.screen.blit(distance_text, (10, 130))
        
        pygame.display.flip()
        
        # 비디오 녹화
        if self.record_video:
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))  # pygame은 (width, height, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)
            
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
        
        for _ in range(3):
            self.spawn_obstacle()
            
        if self.render_mode == "human" or self.record_video:
            self.render()
            
        return self.get_state(), {}

    def step(self, action):
        self.current_step += 1
        self.total_distance += self.speed
        
        # 이벤트 처리 (창 닫기)
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return self.get_state(), 0, True, False, {}
        
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
        
        # 보상 계산
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
        if self.render_mode == "human" or self.record_video:
            self.render()
            
        return self.get_state(), float(reward), bool(terminated), False, {}

    def close(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        if hasattr(self, 'screen'):
            pygame.quit()

# PPO 네트워크 (동일)
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

# PPO 에이전트 (동일하지만 torch.amp.GradScaler 사용)
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, entropy_coef=0.0):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef

        self.network = PPONetwork(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        # FutureWarning 수정
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

    def update(self):
        if len(self.memory) == 0:
            return

        # UserWarning 수정: numpy 배열로 먼저 변환
        states_np = np.array([exp[0] for exp in self.memory])
        states = torch.as_tensor(states_np, dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor([exp[1] for exp in self.memory], dtype=torch.long, device=DEVICE)
        rewards = [exp[2] for exp in self.memory]
        dones = [exp[4] for exp in self.memory]
        old_log_probs = torch.as_tensor([exp[3] for exp in self.memory], dtype=torch.float32, device=DEVICE)

        # 할인 리턴 계산
        discounted_rewards = []
        discounted = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                discounted = 0.0
            discounted = float(r) + self.gamma * discounted
            discounted_rewards.insert(0, discounted)

        returns = torch.as_tensor(discounted_rewards, dtype=torch.float32, device=DEVICE)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        for _ in range(self.k_epochs):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP_ENABLED):
                action_probs, values = self.network(states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                values = values.squeeze(-1)
                advantages = (returns - values).detach()

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, returns)
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.memory.clear()

# 시각화 훈련 함수
def train_with_visualization(
    episodes=1000,
    max_timesteps=2000,
    update_timestep=2000,
    render_mode=None,  # "human" or None
    record_video=False,
    video_episodes=[100, 200, 500, 800],  # 녹화할 에피소드
    seed=42
):
    set_seed(seed)
    
    timestep = 0
    episode_rewards = deque(maxlen=100)
    
    print(f"[Device] {DEVICE.type.upper()} | AMP: {AMP_ENABLED}")
    print("시각화 지오메트리 대시 AI 훈련 시작!")
    
    try:
        pbar = trange(episodes, desc="Episodes", ncols=100)
        for episode in pbar:
            # 특정 에피소드에서만 비디오 녹화
            should_record = record_video and episode in video_episodes
            video_path = f"geometry_dash_episode_{episode}.mp4" if should_record else None
            
            env = VisualGeometryDashEnv(
                render_mode=render_mode if episode % 50 == 0 else None,  # 50 에피소드마다 시각화
                record_video=should_record,
                video_path=video_path
            )
            
            if episode == 0:  # 첫 에피소드에서만 에이전트 생성
                agent = PPOAgent(state_dim=8, action_dim=2)
            
            state, _ = env.reset()
            episode_reward = 0.0

            for t in range(max_timesteps):
                action, log_prob = agent.select_action(state)
                next_state, reward, terminated, _, _ = env.step(action)

                agent.store_experience(state, action, reward, log_prob, terminated)

                state = next_state
                episode_reward += float(reward)
                timestep += 1

                if timestep % update_timestep == 0:
                    agent.update()

                if terminated:
                    break
            
            env.close()
            episode_rewards.append(episode_reward)

            if episode % 50 == 0:
                avg_reward = float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else episode_reward
                pbar.set_postfix(avg_reward=f"{avg_reward:.2f}", last_ep=f"{episode_reward:.2f}")
                
            if should_record:
                print(f"비디오 저장: {video_path}")

        print("훈련 완료!")
        
    except KeyboardInterrupt:
        print("훈련 중단됨")
    
    return agent

# 테스트 함수 (시각화 포함)
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

if __name__ == "__main__":
    # 시각화와 함께 훈련
    trained_agent = train_with_visualization(
        episodes=1000,
        render_mode=None,  # "human"으로 설정하면 실시간 화면 표시
        record_video=True,
        video_episodes=[0, 100, 200, 500, 800],  # 이 에피소드들을 녹화
        seed=42
    )
    
    # 훈련된 모델 테스트 (시각화 포함)
    # test_with_visualization('geometry_dash_ppo_episode_final.pth', episodes=3)


