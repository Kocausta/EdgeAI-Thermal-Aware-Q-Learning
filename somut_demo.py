
"""Sentetik video kareleri -> anomali yogunlugu -> Q-Learning GIF'i."""

from pathlib import Path
import argparse
import csv

from PIL import Image, ImageDraw, ImageFont

from environment import TabularEdgeAIEnv
from main import STATES, train


def font(size):
    paths = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue

    return ImageFont.load_default()


def render(info, state, next_state, before, after,
           i, q_table):

    canvas = Image.new(
        "RGB", (1500, 950), "#f6f8fc"
    )

    d = ImageDraw.Draw(canvas)

    heading = font(23)
    body = font(17)
    small = font(13)

    # BASLIK

    d.text(
        (35, 23),
        "EdgeAI - Sentetik Video Akisi ve Q-Learning",
        fill="#13243a",
        font=heading
    )

    d.text(
        (35, 66),
        f"Adim {i} | Sentetik video simulasyonu",
        fill="#596477",
        font=small
    )

    # AKTIF SENARYO

    d.text(
        (680, 66),
        f"Senaryo: {info['scenario']}",
        fill="#124f72",
        font=body
    )

    # GELEN KARELER

    d.text(
        (35, 110),
        "Gelen kareler: Yesil = Normal, Kirmizi = Anomali",
        fill="#13243a",
        font=body
    )

    labels = info["frame_labels"]

    for j, label in enumerate(labels[:18]):

        x = 35 + (j % 9) * 65
        y = 155 + (j // 9) * 65

        color = "#f7b5b1" if label else "#b4e8c4"

        d.rounded_rectangle(
            (x, y, x + 52, y + 48),
            radius=6,
            fill=color,
            outline="#697c90",
            width=2
        )

        d.text(
            (x + 13, y + 10),
            "A" if label else "N",
            fill="#263444",
            font=body
        )

    # ANOMALI YOGUNLUGU

    d.text(
        (35, 310),
        f"Anomali: {info['anomaly_count']} / "
        f"{info['incoming_frames']} kare",
        fill="#13243a",
        font=body
    )

    d.text(
        (35, 346),
        f"Yogunluk: %{100 * info['anomaly_density']:.1f}",
        fill="#124f72",
        font=body
    )

    # DURUM DEGISIMI

    d.text(
        (35, 389),
        f"Durum: {state} -> Sonraki durum: {next_state}",
        fill="#13243a",
        font=body
    )

    d.text(
        (35, 433),
        f"Q{state}, A{info['action']}: "
        f"{before:.2f} -> {after:.2f}",
        fill="#0e7965",
        font=body
    )

    # AJANIN KARAR PANELI

    d.rounded_rectangle(
        (660, 103, 1460, 500),
        radius=15,
        fill="#e5edf8"
    )

    details = [
        "AJANIN ANLIK KARARI",
        f"Model: {info['model']}",
        f"Cozunurluk: {info['resolution']}",
        f"Kare atlama: {info['frame_skip']}",
        f"Sicaklik: {info['temperature']:.1f} C (simule)",
        f"Kuyruk: {info['queue']:.1f} (simule)",
        f"Odul: {info['reward']:.1f}",
        "Q-degeri bu adimda guncellendi"
    ]

    for j, line in enumerate(details):

        d.text(
            (682, 125 + j * 44),
            line,
            fill="#13243a",
            font=body if j else small
        )

    # Q TABLOSU

    d.text(
        (35, 532),
        "Canli Q-tablosu: 12 durum x 27 eylem",
        fill="#13243a",
        font=heading
    )

    d.text(
        (35, 568),
        "Sari hucre = Bu adimda guncellenen Q-degeri",
        fill="#9b6612",
        font=small
    )

    d.text(
        (20, 601),
        "Durum",
        fill="#334155",
        font=small
    )

    cell_w = 48
    cell_h = 25

    x_start = 165
    y_start = 630

    table_font = font(10)

    # EYLEM BASLIKLARI

    for col in range(27):

        d.text(
            (x_start + col * cell_w + 13, 601),
            f"A{col}",
            fill="#334155",
            font=small
        )

    # Q DEGERLERI

    for row, s in enumerate(STATES):

        y = y_start + row * cell_h

        d.text(
            (29, y + 4),
            f"{s}",
            fill="#334155",
            font=small
        )

        for col, value in enumerate(q_table[row]):

            x = x_start + col * cell_w

            selected = (
                s == state and
                col == info["action"]
            )

            background = (
                "#fde68a" if selected
                else "#e9eff6" if row % 2
                else "white"
            )

            d.rectangle(
                (
                    x,
                    y,
                    x + cell_w - 2,
                    y + cell_h - 2
                ),
                fill=background,
                outline=(
                    "#d97706" if selected
                    else "#ced8e4"
                ),
                width=2 if selected else 1
            )

            d.text(
                (x + 4, y + 6),
                f"{value:.0f}",
                fill="#1e293b",
                font=table_font
            )

    return canvas


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--episodes",
        type=int,
        default=500
    )

    p.add_argument(
        "--steps",
        type=int,
        default=25
    )

    p.add_argument(
        "--output",
        default="demo_results"
    )

    args = p.parse_args()

    if args.episodes < 1 or not 1 <= args.steps <= 200:
        p.error(
            "--episodes en az 1, "
            "--steps 1-200 arasinda olmali"
        )

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True
    )

    # Q-LEARNING EGITIMI

    print("Q-Learning egitimi basliyor...")

    agent, _, _, _ = train(
        args.episodes,
        seed=42,
        record_last=False
    )

    print("Egitim tamamlandi.")

    # SIMULASYON ORTAMI

    env = TabularEdgeAIEnv(
        seed=99,
        max_steps=args.steps
    )

    # DORT FARKLI TEST SENARYOSU

    scenarios = [

        (
            "Normal calisma",
            {
                "temperature": 45,
                "queue": 5,
                "anomaly_rate": 0.20,
                "arrival_rate": 7
            }
        ),

        (
            "Anomali artisi",
            {
                "temperature": 55,
                "queue": 15,
                "anomaly_rate": 0.80,
                "arrival_rate": 9
            }
        ),

        (
            "Asiri isinma",
            {
                "temperature": 85,
                "queue": 10,
                "anomaly_rate": 0.30,
                "arrival_rate": 8
            }
        ),

        (
            "Kuyruk birikmesi",
            {
                "temperature": 65,
                "queue": 70,
                "anomaly_rate": 0.60,
                "arrival_rate": 12
            }
        )

    ]

    current_scenario = -1

    frames = []
    records = []

    # CANLI SIMULASYON ADIMLARI

    for step in range(1, args.steps + 1):

        # DORT SENARYO ARASINDA GECIS

        scenario_index = min(
            3,
            (step - 1) * 4 // args.steps
        )

        if scenario_index != current_scenario:

            current_scenario = scenario_index

            scenario_name, scenario_values = scenarios[
                scenario_index
            ]

            state = env.reset(
                scenario_values
            )

            print(
                f"\nSenaryo basladi: {scenario_name}"
            )

        # AJAN EYLEM SECIYOR

        action = agent.choose_action(
            state,
            explore=False
        )

        # ORTAM GUNCELLENIYOR

        next_state, reward, done, temp = env.step(
            action
        )

        # Q-DEGERI GUNCELLENIYOR

        before, after = agent.learn(
            state,
            action,
            reward,
            next_state,
            done
        )

        info = dict(env.last_info)

        info["scenario"] = scenarios[
            current_scenario
        ][0]

        # GUNCEL Q-TABLOSU

        q_table = [
            agent.get_q(s).copy()
            for s in STATES
        ]

        # ANIMASYON KARESI

        frame = render(
            info,
            state,
            next_state,
            before,
            after,
            step,
            q_table
        )

        frames.append(frame)

        # CSV KAYITLARI

        records.append({

            "step": step,

            "scenario": info["scenario"],

            "normal_count":
                info["incoming_frames"]
                - info["anomaly_count"],

            "anomaly_count":
                info["anomaly_count"],

            "anomaly_density":
                info["anomaly_density"],

            "temperature_simulated":
                info["temperature"],

            "queue_simulated":
                info["queue"],

            "action": action,

            "q_before": before,

            "q_after": after,

            "reward": reward

        })

        state = next_state

    # GIF OLUSTURMA

    gif_path = (
        output / "anlik_anomali_ve_q_learning.gif"
    )

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=800,
        loop=0,
        optimize=False
    )

    # CSV OLUSTURMA

    csv_path = (
        output / "adim_adim_aciklama.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0])
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nTum senaryolar tamamlandi!")

    print(
        f"Toplam adim sayisi: {len(frames)}"
    )

    print(
        f"Sonuclar kaydedildi: {output.resolve()}"
    )

    print(
        "Not: Anomali etiketleri sentetiktir; "
        "termal veriler simule edilmistir."
    )


if __name__ == "__main__":
    main()
