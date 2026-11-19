
"""Eğitim, karşılaştırma, grafikler ve Q-tablosu animasyonu."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from agent import TabularQLearningAgent
from environment import TabularEdgeAIEnv


STATES = [
    (t, q, a)
    for t in range(3)
    for q in range(2)
    for a in range(2)
]


class LiveQTable:
    """Son eğitim bölümünü canlı gösteren Q-tablosu."""

    def __init__(self):

        self.enabled = "agg" not in plt.get_backend().lower()

        if not self.enabled:
            print("Canlı pencere kullanılamıyor. GIF ve eğitim devam ediyor.")
            return

        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(17, 8))

        self.fig.subplots_adjust(
            left=0.09,
            right=0.99,
            top=0.77,
            bottom=0.07
        )

        self.ax.axis("off")

        self.table = self.ax.table(
            cellText=[["0.0"] * 27 for _ in STATES],
            colLabels=[f"A{i}" for i in range(27)],
            rowLabels=[str(s) for s in STATES],
            cellLoc="center",
            rowLoc="center",
            bbox=[0, 0, 1, 1]
        )

        self.table.auto_set_font_size(False)
        self.table.set_fontsize(7)

        for (r, c), cell in self.table.get_celld().items():

            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.4)

            if r == 0:
                cell.set_facecolor("#377bd1")
                cell.get_text().set_color("white")

            elif c == -1:
                cell.set_facecolor("#e2e8f0")

        self.fig.suptitle(
            "CANLI Q-LEARNING | 12 Durum x 27 Eylem",
            fontsize=17
        )

        self.details = self.fig.text(0.06, 0.91, "", fontsize=11)
        self.action_text = self.fig.text(0.06, 0.86, "", fontsize=11)
        self.update_text = self.fig.text(
            0.06, 0.81, "",
            fontsize=11,
            color="#0f766e"
        )

        self.previous = None

        plt.show(block=False)

    def update(self, agent, state, action, before, after, info):

        if not self.enabled:
            return

        if not plt.fignum_exists(self.fig.number):
            self.enabled = False
            return

        if self.previous is not None:
            self.table[self.previous].set_facecolor("white")

        for r, s in enumerate(STATES, 1):

            for c, value in enumerate(agent.get_q(s)):

                self.table[r, c].get_text().set_text(
                    f"{value:.1f}"
                )

        selected = (
            STATES.index(state) + 1,
            action
        )

        self.table[selected].set_facecolor("#fde68a")
        self.previous = selected

        self.details.set_text(
            f"Adım {info['step']}/200 | "
            f"Sıcaklık: {info['temperature']:.1f}°C | "
            f"Kuyruk: {info['queue']:.1f} | "
            f"Anomali: %{100 * info['anomaly_density']:.1f}"
        )

        self.action_text.set_text(
            f"A{action}: {info['model']} | "
            f"Çözünürlük: {info['resolution']} | "
            f"Kare atlama: {info['frame_skip']} | "
            f"Ödül: {info['reward']:.2f}"
        )

        self.update_text.set_text(
            f"Q{state}, A{action}: "
            f"{before:.2f} -> {after:.2f} "
            f"(Değişim: {after-before:+.2f})"
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def finish(self):

        if self.enabled and plt.fignum_exists(self.fig.number):

            print("Canlı eğitim tamamlandı. Pencereyi kapatabilirsiniz.")

            plt.ioff()
            plt.show(block=True)


# =====================================================
# Q-LEARNING EGITIMI
# =====================================================

def train(episodes, seed=42, record_last=True, live=False):

    env = TabularEdgeAIEnv(seed=seed)

    agent = TabularQLearningAgent(
        env.action_space_n,
        seed=seed
    )

    episode_rows = []
    step_rows = []
    snapshots = []

    live_table = None

    for ep in range(episodes):

        if live and ep == episodes - 1:
            live_table = LiveQTable()

        state = env.reset()

        total_reward = 0.0
        max_temp = env.temp
        critical_steps = 0

        queue_sum = 0.0
        latency_sum = 0.0
        density_sum = 0.0

        for step in range(env.max_steps):

            action = agent.choose_action(state)

            next_state, reward, done, temp = env.step(action)

            before, after = agent.learn(
                state,
                action,
                reward,
                next_state,
                done
            )

            total_reward += reward
            max_temp = max(max_temp, temp)

            critical_steps += temp > 70

            queue_sum += env.queue_len
            latency_sum += env.last_info["latency"]
            density_sum += env.anomaly_density

            if record_last and ep == episodes - 1:

                step_rows.append({
                    "episode": ep + 1,
                    "state": str(state),
                    "next_state": str(next_state),
                    "q_before": before,
                    "q_after": after,
                    "q_change": after - before,
                    **env.last_info
                })

                snapshots.append({
                    "matrix": np.array([
                        agent.get_q(s).copy()
                        for s in STATES
                    ]),
                    "state": state,
                    "action": action,
                    "before": before,
                    "after": after,
                    "info": dict(env.last_info)
                })

                if live_table is not None:

                    live_table.update(
                        agent,
                        state,
                        action,
                        before,
                        after,
                        env.last_info
                    )

            state = next_state

            if done:
                break

        episode_rows.append({

            "episode": ep + 1,

            "reward": total_reward,

            "max_temperature": max_temp,

            "critical_steps": critical_steps,

            "mean_queue": queue_sum / env.max_steps,

            "mean_latency": latency_sum / env.max_steps,

            "mean_anomaly_density": density_sum / env.max_steps,

            "epsilon": agent.epsilon

        })

        agent.end_episode()

    if live_table is not None:
        live_table.finish()

    return (
        agent,
        pd.DataFrame(episode_rows),
        pd.DataFrame(step_rows),
        snapshots
    )


# =====================================================
# EGITILMIS AJANI TEST ETME
# =====================================================

def evaluate(agent, seed, episodes=50):

    env = TabularEdgeAIEnv(seed=seed)

    results = []

    for ep in range(episodes):

        state = env.reset()

        total = 0.0
        critical = 0

        latency_sum = 0.0
        anomaly_sum = 0.0
        queue_sum = 0.0

        max_temp = env.temp

        for _ in range(env.max_steps):

            action = agent.choose_action(
                state,
                explore=False
            )

            state, reward, done, temp = env.step(action)

            total += reward

            critical += temp > 70

            latency_sum += env.last_info["latency"]

            anomaly_sum += env.last_info["anomaly_density"]

            queue_sum += env.queue_len

            max_temp = max(max_temp, temp)

            if done:
                break

        results.append({

            "reward": total,

            "critical_step_rate":
                critical / env.max_steps,

            "mean_latency":
                latency_sum / env.max_steps,

            "mean_queue":
                queue_sum / env.max_steps,

            "max_temperature":
                max_temp,

            "mean_anomaly_density":
                anomaly_sum / env.max_steps

        })

    return pd.DataFrame(results).mean(
        numeric_only=True
    ).to_dict()


# =====================================================
# Q-TABLOSU GIF ANIMASYONU
# =====================================================

def draw_heatmap(snapshots, output, every=5):

    indices = list(
        range(0, len(snapshots), every)
    )

    if indices[-1] != len(snapshots) - 1:
        indices.append(len(snapshots) - 1)

    fig, ax = plt.subplots(figsize=(21, 10))

    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.74,
        bottom=0.09
    )

    ax.axis("off")

    labels = [f"A{i}" for i in range(27)]

    grid = [
        [f"{v:.1f}" for v in row]
        for row in snapshots[0]["matrix"]
    ]

    table = ax.table(
        cellText=grid,
        colLabels=labels,
        rowLabels=[str(s) for s in STATES],
        loc="center",
        cellLoc="center",
        rowLoc="center",
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)

    for (r, c), cell in table.get_celld().items():

        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.45)

        if r == 0:
            cell.set_facecolor("#377bd1")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")

        elif c == -1:
            cell.set_facecolor("#e2e8f0")
            cell.get_text().set_weight("bold")

    fig.text(
        0.035,
        0.96,
        "Q-TABLOSU | 12 Durum x 27 Eylem",
        fontsize=19,
        weight="bold"
    )

    details = fig.text(
        0.035, 0.915, "",
        fontsize=11,
        color="#1e293b"
    )

    action_details = fig.text(
        0.035, 0.875, "",
        fontsize=11,
        color="#1e293b"
    )

    change_details = fig.text(
        0.035, 0.835, "",
        fontsize=12,
        weight="bold",
        color="#0f766e"
    )

    fig.text(
        0.035,
        0.785,
        "Durum = (Sıcaklık, Kuyruk, Anomali) | "
        "Sarı hücre = Güncellenen Q-değeri",
        fontsize=10,
        color="#475569"
    )

    previous = [None]

    def update(i):

        snap = snapshots[indices[i]]

        state = snap["state"]
        action = snap["action"]
        info = snap["info"]

        row = STATES.index(state) + 1

        if previous[0] is not None:

            table[previous[0]].set_facecolor("#ffffff")

            table[previous[0]].set_edgecolor("#cbd5e1")

            table[previous[0]].set_linewidth(0.45)

        for r in range(12):

            for c in range(27):

                table[r + 1, c].get_text().set_text(
                    f"{snap['matrix'][r, c]:.1f}"
                )

        table[row, action].set_facecolor("#fde68a")

        table[row, action].set_edgecolor("#d97706")

        table[row, action].set_linewidth(2.2)

        previous[0] = (row, action)

        details.set_text(
            f"Son eğitim bölümü | "
            f"Adım {indices[i] + 1}/{len(snapshots)} | "
            f"Sıcaklık: {info['temperature']:.1f}°C | "
            f"Kuyruk: {info['queue']:.1f} | "
            f"Anomali: %{100 * info['anomaly_density']:.1f}"
        )

        action_details.set_text(
            f"A{action} | "
            f"Model: {info['model']} | "
            f"Çözünürlük: {info['resolution']} | "
            f"Kare atlama: {info['frame_skip']} | "
            f"Ödül: {info['reward']:.2f}"
        )

        change_details.set_text(
            f"Q{state}, A{action}: "
            f"{snap['before']:.2f} -> "
            f"{snap['after']:.2f} "
            f"(Değişim: "
            f"{snap['after'] - snap['before']:+.2f})"
        )

        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=130,
        blit=False
    )

    animation.save(
        output,
        writer=PillowWriter(fps=5),
        dpi=85
    )

    plt.close(fig)


# =====================================================
# ANA PROGRAM
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    # EGITIM SAYISI ARTIRILDI

    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 3000]
    )

    parser.add_argument(
        "--output",
        default="results"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--test-episodes",
        type=int,
        default=50
    )

    parser.add_argument(
        "--no-gif",
        action="store_true"
    )

    parser.add_argument(
        "--live",
        action="store_true"
    )

    args = parser.parse_args()

    if (
        any(n < 1 for n in args.episodes)
        or args.test_episodes < 1
    ):
        parser.error("Bölüm sayıları pozitif olmalı.")

    output = Path(args.output)

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    comparisons = []

    # FARKLI EGITIM BUYUKLUKLERI

    for episodes in args.episodes:

        print(f"Eğitim: {episodes} bölüm...")

        agent, learning, steps, snapshots = train(
            episodes,
            seed=args.seed,
            live=(
                args.live
                and episodes == args.episodes[-1]
            )
        )

        metrics = evaluate(
            agent,
            seed=args.seed + 10000,
            episodes=args.test_episodes
        )

        comparisons.append({
            "episodes": episodes,
            **metrics
        })

        learning.to_csv(
            output / f"egitim_{episodes}.csv",
            index=False
        )

        if episodes == args.episodes[-1]:

            steps.to_csv(
                output / "anlik_adim_degerleri.csv",
                index=False
            )

            np.save(
                output / "q_tablosu.npy",
                np.array([
                    agent.get_q(s)
                    for s in STATES
                ])
            )

            if not args.no_gif:

                draw_heatmap(
                    snapshots,
                    output / "q_tablosu_animasyon.gif"
                )

            # ODUL GRAFIGI

            plt.figure(figsize=(9, 4))

            plt.plot(
                learning.episode,
                learning.reward,
                alpha=0.3,
                label="Bölüm ödülü"
            )

            plt.plot(
                learning.episode,
                learning.reward.rolling(
                    50,
                    min_periods=1
                ).mean(),
                label="50 bölümlük ortalama"
            )

            plt.xlabel("Bölüm")
            plt.ylabel("Toplam ödül")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            plt.savefig(
                output / "ogrenme_grafigi.png",
                dpi=160
            )

            plt.close()

    # =================================================
    # EGITIM KARSILASTIRMASI
    # =================================================

    comparison = pd.DataFrame(comparisons)

    comparison.to_csv(
        output / "egitim_karsilastirmasi.csv",
        index=False
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8)
    )

    plots = [
        ("reward", "Ortalama Toplam Ödül"),
        ("critical_step_rate", "Kritik Sıcaklık Oranı"),
        ("mean_latency", "Ortalama Gecikme"),
        ("mean_queue", "Ortalama Kuyruk Uzunluğu")
    ]

    for ax, (key, title) in zip(
        axes.flat,
        plots
    ):

        ax.plot(
            comparison["episodes"],
            comparison[key],
            marker="o",
            linewidth=2
        )

        ax.set_title(title)
        ax.set_xlabel("Eğitim Bölümü")

        ax.set_xticks(
            comparison["episodes"]
        )

        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Bağımsız Test: {args.test_episodes} Bölüm"
    )

    fig.tight_layout()

    fig.savefig(
        output / "performans_karsilastirmasi.png",
        dpi=180
    )

    plt.close(fig)

    # =================================================
    # DETAYLI EGITIM GRAFIKLERI
    # =================================================

    last_learning = learning

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8)
    )

    training_plots = [
        ("reward", "Eğitim Ödülü"),
        ("max_temperature", "Maksimum Sıcaklık"),
        ("mean_queue", "Ortalama Kuyruk"),
        ("mean_latency", "Ortalama Gecikme")
    ]

    for ax, (key, title) in zip(
        axes.flat,
        training_plots
    ):

        ax.plot(
            last_learning.episode,
            last_learning[key],
            alpha=0.2,
            linewidth=0.9
        )

        ax.plot(
            last_learning.episode,
            last_learning[key].rolling(
                50,
                min_periods=1
            ).mean(),
            linewidth=1.6,
            label="50 bölümlük ortalama"
        )

        ax.set_title(title)
        ax.set_xlabel("Bölüm")
        ax.grid(alpha=0.3)
        ax.legend()

    # KRITIK SICAKLIK SINIRI

    axes[0, 1].axhline(
        70,
        color="red",
        linestyle="--",
        label="Kritik Sıcaklık: 70°C"
    )

    axes[0, 1].legend()

    fig.tight_layout()

    fig.savefig(
        output / "egitim_detay_grafikleri.png",
        dpi=180
    )

    plt.close(fig)

    # SONUCLAR

    print("\n--- TEST SONUÇLARI ---")

    print(
        comparison.round(3).to_string(
            index=False
        )
    )

    print(
        f"\nDosyalar kaydedildi: {output.resolve()}"
    )


if __name__ == "__main__":
    main()
