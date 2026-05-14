import pandas as pd
import numpy as np

def generate_diabetes_dataset(n=500, noise_ratio=0.1, seed=42):
    np.random.seed(seed)

    half = n // 2
    noise_n = int(n * noise_ratio / 2)

    # ======================
    # Healthy (0)
    # ======================
    healthy = pd.DataFrame({
        "blood_sugar": np.random.randint(70, 125, half),
        "blood_pressure": np.random.randint(65, 90, half),
        "diabetes": 0
    })

    # ======================
    # Diabetic (1)
    # ======================
    diabetic = pd.DataFrame({
        "blood_sugar": np.random.randint(130, 220, half),
        "blood_pressure": np.random.randint(80, 125, half),
        "diabetes": 1
    })

    # ======================
    # Noise (mixing the classes)
    # ======================
    healthy_noise = pd.DataFrame({
        "blood_sugar": np.random.randint(120, 145, noise_n),
        "blood_pressure": np.random.randint(80, 100, noise_n),
        "diabetes": 0
    })

    diabetic_noise = pd.DataFrame({
        "blood_sugar": np.random.randint(105, 135, noise_n),
        "blood_pressure": np.random.randint(70, 95, noise_n),
        "diabetes": 1
    })

    # ======================
    # Combine
    # ======================
    df = pd.concat(
        [healthy, diabetic, healthy_noise, diabetic_noise],
        ignore_index=True
    )

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df

df = generate_diabetes_dataset()
