# Social Grid Simulation: Emergent Dynamics of Confidence, Beauty, and Social Matching

## Abstract
This simulation models a stylized social environment populated by men and women, each with evolving attributes and daily routines.  
Using **agent-based modeling** and reinforcement-style rules, individuals seek partners in a shared café environment.  
Acceptance and rejection dynamics shape **confidence, beauty, reproduction, and mortality**.  

Over long runs, the system exhibits **population booms, collapses, and eventual extinction** — driven by feedback between attributes and demographic pressures.

---

## 🎥 Preview Video

[![Watch the demo](https://img.youtube.com/vi/gadEXRTXBAU/hqdefault.jpg)](https://youtu.be/gadEXRTXBAU)

The video shows:
- Agents moving between **Homes**, **Work**, and the **Café**
- Accept/reject dynamics shaping confidence and beauty
- Birth and death events tracked live
- How the population eventually collapses to extinction




## 1. Introduction
Understanding emergent patterns of human social interactions is a core interest of **computational social science**.  
This project implements a grid-based simulator in **Python + Pygame**, with agents moving between **homes, workplaces, and cafés**.  

### Key Rules
- **Men initiate** interactions, women evaluate.  
- **Women accept only the top X%** of men by composite score.  
- **Men accept women if beauty ≥ threshold**.  
- **Confidence** (men) and **beauty** (women) update dynamically with success/failure.  
- **Births** occur probabilistically after repeated successes.  
- **Deaths** occur by aging, isolation, or beauty/confidence collapse.

![](/Simulation/Dashboard1.png)

---

## 2. Methods

### 2.1 Environment
- Grid size: **1920×1080**, 70×54 cells.  
- Buildings:
  - Left blocks: **Men’s homes**  
  - Right blocks: **Women’s homes**  
  - Center: **Work (upper)** and **Café (lower)**  

![](/Simulation/Dashboard_empty.png)

Daily cycle:
- Sleep: 01:30–08:30  
- Work: 09:00–17:00 (morning) or 17:00–01:00 (afternoon)  
- Café/home: discretionary hours, modulated by confidence/sociality.  

### 2.2 Agents
- **Men**: money, power, health, confidence → composite score.  
- **Women**: beauty (decays if no matches, recovers with success), sociality proxy.  

![](/Simulation/work.png)
![](/Simulation/Cafe.png)

### 2.3 Dynamics
- **Interaction**: Café encounters; women accept if man’s score ≥ top-X% threshold.  
- **Feedback**:
  - Confidence ↑ after acceptance, ↓ after rejection.  
  - Beauty ↑ after acceptance, ↓ after repeated no-match days.  
- **Births**: After n accepts, subject to population pressure.  
- **Deaths**:  
  - Isolation (low confidence/sociality).  
  - Beauty ≤ 0.05 (lethal for women).  
  - Natural mortality.  

---

## 3. Results

### 3.1 Population Dynamics
- Early growth due to matches and births.  
- Mid-simulation: confidence collapse + beauty decay triggered isolation spirals.  
- Late-simulation: **entire population died out**.  

![](/Simulation/Charts.png)

### 3.2 Attribute Feedback
- **Confidence collapse**: Men locked out after rejection loops.  
- **Beauty fragility**: Women without matches decayed to lethal thresholds.  
- **Top-X% filter**: A small male elite monopolized success.  

### 3.3 Emergent Behaviors
- **Self-isolation loops**: Agents retreated to homes until death.  
- **Boom-bust cycles**: Birth surges followed by collapse.  
- **Elite lock-in**: Few agents accumulated nearly all accepts.  

![](/Simulation/RandomManStats.png)
![](/Simulation/RandomWomanStats.png)

---

## 4. Dashboard & Analytics

The simulator includes a **real-time dashboard** with multiple panels:

- **Summary**: Key simulation parameters & daily stats.  
- **Selected**: Detailed view of an agent’s history and outcomes.  
- **Charts**: Longitudinal plots of population, births, and deaths (by gender).  
- **Leaderboards**: Ranking men by score, women by beauty.  

![](/Simulation/Summary.png)
![](/Simulation/Leaderboards.png)

---

## 5. Discussion
The simulation shows how **simple local rules** generate **systemic demographic collapse**:  

- Selectivity (top-X% rule) drives inequality.  
- Confidence/beauty feedback loops magnify failure.  
- Birth pressure is insufficient to offset cascading isolation.  

### Key Insights
- **Threshold choice** is critical in shaping systemic stability.  
- Positive/negative feedback loops create **boom-bust cycles**.  
- Social selectivity may lead to **population collapse** in closed systems.  

---

## 6. Conclusion
The Social Grid Simulator acts as a sandbox for emergent social science:  

- Models inequality, thresholds, and feedback dynamics.  
- Visualized in a **professional dark dashboard**.  
- Logs **CSV + events** for quantitative analysis.  
- In this experiment, the **entire population went extinct** after ~200 days.  

---

## 7. Repository Notes
- Language: **Python 3.10+**  
- Engine: **Pygame 2.6**  
- Single-file implementation (`main.py`).  
- Hotkeys documented in the Help panel.  

