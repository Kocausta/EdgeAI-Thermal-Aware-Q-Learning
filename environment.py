"""Sentetik video akışı üzerinde termal duyarlı EdgeAI simülasyonu."""
import numpy as np


class TabularEdgeAIEnv:
    def __init__(self, seed=None, max_steps=200):
        self.rng = np.random.default_rng(seed)
        self.frame_skips = [1, 2, 4]
        self.model_sizes = [0, 1, 2]
        self.resolutions = [0.5, 0.75, 1.0]
        self.actions = [(fs, ms, res) for fs in self.frame_skips
                        for ms in self.model_sizes for res in self.resolutions]
        self.action_space_n = len(self.actions)
        self.max_steps = max_steps
        self.reset()

    def reset(self, scenario=None):
        scenario = scenario or {}
        self.temp = float(scenario.get("temperature", self.rng.uniform(40, 65)))
        self.queue_len = float(scenario.get("queue", self.rng.uniform(0, 25)))
        self.base_anomaly_rate = float(scenario.get("anomaly_rate", self.rng.uniform(0.05, 0.65)))
        self.arrival_rate = float(scenario.get("arrival_rate", self.rng.uniform(5, 12)))
        self.anomaly_density = self.base_anomaly_rate
        self.step_count = 0
        self.last_info = {}
        return self._get_discretized_state()

    def _get_discretized_state(self):
        t_bin = 0 if self.temp < 60 else (1 if self.temp < 80 else 2)
        q_bin = 0 if self.queue_len < 20 else 1
        a_bin = 0 if self.anomaly_density < 0.5 else 1
        return (t_bin, q_bin, a_bin)

    def step(self, action_idx, observed_anomaly_density=None):
        self.step_count += 1
        fs, ms, res = self.actions[action_idx]
        compute_load = res ** 2 * [0.3, 0.6, 1.2][ms] / fs
        self.temp = float(np.clip(self.temp + compute_load * 3.5 - 0.8, 30, 95))

        # Her adımda sentetik kare etiketleri üretilir: 1=anomali, 0=normal.
        incoming = int(self.rng.poisson(self.arrival_rate))
        self.base_anomaly_rate = float(np.clip(
            self.base_anomaly_rate + self.rng.normal(0, 0.035), 0.02, 0.95))
        labels = self.rng.binomial(1, self.base_anomaly_rate, size=incoming)
        if observed_anomaly_density is None:
            self.anomaly_density = float(labels.mean()) if incoming else self.base_anomaly_rate
        else:
            # Gerçek görüntü modelinin gözlemi: simüle etiketler kullanılmaz.
            self.anomaly_density = float(np.clip(observed_anomaly_density, 0, 1))

        processed_capacity = 10.0 / fs
        self.queue_len = float(np.clip(self.queue_len + incoming - processed_capacity, 0, 100))
        latency = self.queue_len * 5 + compute_load * 40
        accuracy_proxy = 0.5 * res + 0.5 * [0.7, 0.85, 1.0][ms]
        processed_ratio = 1.0 / fs
        expected_captured = self.anomaly_density * accuracy_proxy * processed_ratio
        reward_anomaly = expected_captured * 100
        reward_latency = -max(0, latency - 100) * 0.1
        # Üstel ceza uzun sıcak bölümlerde ödülü aşırı negatif yapıyordu.
        # Karesel ceza ısınmayı güçlü biçimde cezalandırır fakat patlamaz.
        reward_thermal = -2.0 * (self.temp - 70) ** 2 if self.temp > 70 else 0.0
        reward = float(reward_anomaly + reward_latency + reward_thermal)
        done = self.step_count >= self.max_steps
        self.last_info = {
            "step": self.step_count, "temperature": self.temp, "queue": self.queue_len,
            "anomaly_density": self.anomaly_density, "anomaly_count": int(labels.sum()),
            "frame_labels": labels.astype(int).tolist(),
            "anomaly_source": "synthetic" if observed_anomaly_density is None else "visual_model",
            "incoming_frames": incoming, "frame_skip": fs, "model": ["Nano", "Small", "Medium"][ms],
            "resolution": res, "latency": float(latency), "reward": reward,
            "reward_anomaly": float(reward_anomaly), "reward_latency": float(reward_latency),
            "reward_thermal": float(reward_thermal), "action": action_idx,
        }
        return self._get_discretized_state(), reward, done, self.temp
