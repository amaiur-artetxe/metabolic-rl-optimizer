import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class GlucoseEnvMultiAction(gym.Env):
    """
    Reinforcement Learning environment for adaptive glucose regulation in T1DM.

    The supervised predictive model acts as a physiological simulator.
    The agent learns an intervention policy to maintain glucose in a safe range.

    ACTION SPACE:
        action[0] -> additional physical activity (steps in next 30 min)
        action[1] -> carbohydrate servings (1 serving = 10 grams)

    SERVING DEFINITION:
        1 carbohydrate serving = 10 g (as defined in the dataset)

    OBJECTIVE:
        - Keep glucose near 110 mg/dL
        - Maximize Time-In-Range (70–180 mg/dL)
        - Avoid hypoglycemia
        - Avoid unnecessary interventions

    Research-use only. Not medical advice.
    """

    def __init__(self, df_model, model):
        super().__init__()

        self.df_original = df_model.copy().reset_index(drop=True)
        self.model = model

        self.state_columns = [
            "glucose_lag1",
            "glucose_lag2",
            "glucose_ma_30",
            "carbs_45m",
            "bolus_45m",
            "steps_30m",
            "calories_30m",
            "heart_rate_ma_30",
            "basal_rate_45m",
            "hour"
        ]

        # Scale state features
        self.scaler = StandardScaler()
        self.scaler.fit(self.df_original[self.state_columns])

        # ACTION SPACE
        # steps: 0–2000
        # carb servings: 0–6 (0–60 g)

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([2000.0, 6.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state_columns),),
            dtype=np.float32
        )

        self.max_steps = len(self.df_original) - 2
        self.reset()

    # --------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.df = self.df_original.copy()
        self.current_step = 0

        return self._get_state(), {}

    # --------------------------------------------------

    def _get_state(self):
        state_df = self.df.loc[[self.current_step], self.state_columns]
        state = self.scaler.transform(state_df).flatten()
        return state.astype(np.float32)

    # --------------------------------------------------

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)

        steps = float(action[0])
        carb_servings = float(action[1])

        # Convert servings to grams
        carbs_grams = carb_servings * 10.0

        row = self.df.loc[self.current_step, self.state_columns].copy()

        # Apply physical activity
        row["steps_30m"] += steps
        row["calories_30m"] += steps * 0.04
        row["heart_rate_ma_30"] += steps * 0.005

        # Apply carbohydrate intake
        row["carbs_45m"] += carbs_grams

        # Predict next glucose
        X = pd.DataFrame([row.values], columns=self.state_columns)
        predicted_glucose = self.model.predict(X)[0]

        # ---------------- REWARD ----------------

        target = 110
        reward = -((predicted_glucose - target) ** 2) / 100

        if predicted_glucose < 70:
            reward -= 100

        elif predicted_glucose > 250:
            reward -= 40

        # Penalize unnecessary interventions
        reward -= steps * 0.0003
        reward -= carb_servings * 0.3

        # ---------------- DYNAMICS UPDATE ----------------

        if self.current_step + 1 < self.max_steps:

            self.df.loc[self.current_step + 1, "glucose_lag2"] = \
                self.df.loc[self.current_step, "glucose_lag1"]

            self.df.loc[self.current_step + 1, "glucose_lag1"] = predicted_glucose

            prev_ma = self.df.loc[self.current_step, "glucose_ma_30"]
            new_ma = (prev_ma * 5 + predicted_glucose) / 6
            self.df.loc[self.current_step + 1, "glucose_ma_30"] = new_ma

        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        next_state = self._get_state()

        info = {
            "predicted_glucose": predicted_glucose,
            "steps_action": steps,
            "carb_servings_action": carb_servings
        }

        return next_state, reward, terminated, truncated, info