# metabolic-rl-optimizer  

Reinforcement Learning for Glucose Dynamics Optimization using real-world Type 1 Diabetes data.

---

## 📌 Project Overview  

**metabolic-rl-optimizer** is a model-based Reinforcement Learning framework designed to optimize glucose dynamics in individuals with Type 1 Diabetes (T1D) using real-world continuous glucose monitoring (CGM) data.

The system combines:

- Patient selection and data-driven filtering  
- Exploratory Data Analysis (EDA)  
- Advanced feature engineering  
- Glucose prediction using XGBoost  
- Model-based Reinforcement Learning using Soft Actor-Critic (SAC)  

The objective is to learn an optimal intervention policy that maintains glucose levels in a safe physiological range while minimizing unnecessary interventions.

---

## 🧪 1. Patient Selection & Data Exploration  

A careful selection of the optimal patient for the RL setting was performed based on:

- Data completeness  
- Stability of recordings  
- Sufficient variability in glucose, activity and nutrition  

Exploratory Data Analysis included:

- Glucose distribution analysis  
- Time-in-range statistics  
- Correlation between glucose, physical activity and carbohydrates  
- Temporal behavior patterns  

---

## 🧠 2. Feature Engineering  

Features were designed to capture short-term physiological dynamics:

### Glucose dynamics
- `glucose_lag1`
- `glucose_lag2`
- `glucose_ma_30` (30-min moving average)

### Nutrition (last 45 minutes)
- `carbs_45m`
- `bolus_45m`
- `basal_rate_45m`

### Physical activity (last 30 minutes)
- `steps_30m`
- `calories_30m`
- `heart_rate_ma_30`

### Temporal context
- `hour`

Carbohydrate servings were defined as:

> 1 serving = 10 grams of carbohydrates

---

The predictive model enables a **model-based RL approach**, where transitions are generated through learned physiological dynamics rather than a black-box simulator.

---

## 🤖 4. Reinforcement Learning (Model-Based)

### Algorithm

The RL agent was trained using:

- **Soft Actor-Critic (SAC)**  
- Continuous action space  
- Model-based environment  

---

## 🏥 RL Environment Design

The environment was implemented using `gymnasium`.

### 🎯 Objective

The agent learns a policy to:

- Maintain glucose near 110 mg/dL  
- Maximize Time-In-Range (70–180 mg/dL)  
- Avoid hypoglycemia  
- Avoid excessive interventions  

---

### 🧾 State Space

Scaled continuous state including:

- Glucose lags and moving average  
- Carbohydrates (last 45 min)  
- Insulin (bolus + basal)  
- Physical activity metrics (last 30 min)  
- Hour of day  

---

### 🎮 Action Space (Continuous, 2D)

action[0] → Additional physical activity (steps in next 30 minutes)
action[1] → Carbohydrate servings (servings in next 45 minutes, 1 serving = 10g)

Constraints:

- Steps: 0 – 2000  
- Carbs: 0 – 60 grams  

The agent decides **activity and carbohydrate intake interventions**.

---

### 🏆 Reward Function

The reward is shaped to enforce physiological safety:

- Quadratic penalty from target (110 mg/dL)
- Strong penalty for hypoglycemia (<70 mg/dL)
- Penalty for severe hyperglycemia (>250 mg/dL)
- Regularization penalty for excessive steps
- Regularization penalty for carbohydrate intake

This encourages stability while discouraging unnecessary interventions.

---

## 🔄 Model-Based Transition Dynamics

At each step:

1. The agent proposes activity and carbohydrate interventions.
2. The environment updates:
   - steps → calories burned & heart rate
   - carb servings → grams intake
3. The XGBoost model predicts next glucose.
4. Lag features and moving averages are updated.
5. The next state is generated.

The predictive model acts as a learned physiological simulator.

---

## 📊 Evaluation Metrics

Policy evaluation includes:

- Time in Range (70–180 mg/dL)
- Time Below Range (<70 mg/dL)
- Time Above Range (>180 mg/dL)
- Predicted glucose trajectories
- Comparison vs real patient behavior



