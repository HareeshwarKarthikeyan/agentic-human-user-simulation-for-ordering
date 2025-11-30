---
theme: academic
background: https://cover.sli.dev
title: Multi-Agent Framework for Human User Simulation
info: |
  ## Multi-Agent Framework for Human User Simulation in Interactive Conversational AI Systems
  
  A novel approach to testing conversational AI at scale through specialized agent decomposition
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Multi-Agent Framework for Human User Simulation

## in Interactive Conversational AI Systems

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    A Novel Approach to Testing Conversational AI at Scale <carbon:arrow-right class="inline"/>
  </span>
</div>

<div class="pt-8">
  <a href="https://drive.google.com/file/d/15uC9aVw9isTHucUy8r2KdENEdqU2bfVB/view?usp=sharing" target="_blank" class="text-blue-400 hover:text-blue-300 underline">
    📄 Read the Full Paper
  </a>
</div>

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.toasttab.com/hareeshkarthik-toast/agentic-human-guest-simulation-for-ordering" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

---
theme: academic
---

# The Challenge

<div class="flex flex-col gap-4">

<div class="w-full">

![Intro Figure](/paper/draft/Intro%20Figure.jpg)

</div>

<div class="grid grid-cols-2 gap-4">

<div>

**Testing conversational AI at scale is difficult**
- Need realistic, diverse user interactions
- Complex behavioral patterns across personas
- Multi-turn conversation dynamics
- Scalable and reproducible testing

</div>

<div>

**Current approaches fall short**
- Static test sets miss dynamic nature
- Human evaluators: expensive & hard to scale
- Single LLM: lacks behavioral diversity
- Poor explainability and control

</div>

</div>

</div>

---
theme: academic
layout: center
class: text-center
---

# Our Solution

<div class="text-6xl text-primary mb-8">
  Multi-Agent Orchestration
</div>

<div class="grid grid-cols-3 gap-8 mt-12">
  <div class="bg-gradient-to-br from-blue-500/20 to-blue-600/20 p-6 rounded-lg">
    <div class="text-2xl mb-4">🤖</div>
    <h3 class="text-xl font-bold mb-2">User</h3>
    <p class="text-sm mb-2">Primary orchestrator</p>
    <p class="text-xs">Generates contextually appropriate responses based on input messages, state, and attributes</p>
  </div>
  <div class="bg-gradient-to-br from-green-500/20 to-green-600/20 p-6 rounded-lg">
    <div class="text-2xl mb-4">📊</div>
    <h3 class="text-xl font-bold mb-2">State Tracking</h3>
    <p class="text-sm mb-2">Structured task state</p>
    <p class="text-xs">Maintains current progress and target goals throughout the conversation</p>
  </div>
  <div class="bg-gradient-to-br from-purple-500/20 to-purple-600/20 p-6 rounded-lg">
    <div class="text-2xl mb-4">🎭</div>
    <h3 class="text-xl font-bold mb-2">Message Attributes</h3>
    <p class="text-sm mb-2">Behavioral control</p>
    <p class="text-xs">Controls mood, execution style, and exploration patterns</p>
  </div>
</div>

---
theme: academic
---

# Agentic Setup Methodology

<div class="grid grid-cols-2 gap-4 text-xs">

<div>

<v-clicks>

### User Response Generation

$$r_t = f_{user}(m_t, s_t, a_t)$$

- $r_t$ = response at turn $t$
- $m_t$ = input message
- $s_t$ = task state
- $a_t$ = behavioral attributes

### State Tracking

$$s_t = f_{stateTracking}(input\_message)$$
$$= \{\mathcal{T}_{current}, \mathcal{T}_{target}\}$$

- $\mathcal{T}_{current}$ = confirmed task items
- $\mathcal{T}_{target}$ = desired final state

</v-clicks>

</div>

<div>

<v-clicks>

### Message Attributes

$$a_t = f_{msgAttrGen}(p_{bio}, s_t)$$

$$a_t = \{mood\_tone, task\_execution\_style,$$
$$exploration\_style, task\_completion\_status\}$$

- $p_{bio}$ = persona biography
- $mood\_tone \in${casual, frustrated, confused, enthusiastic}
- $task\_execution\_style \in${one-by-one, all-at-once}
- $exploration\_style \in${explores, does-not-explore}
- $task\_completion\_status \in${complete, incomplete}

### Exit Gating

$$task\_completion\_status = \begin{cases}
complete & \text{if } \mathcal{T}_{current} \supseteq \mathcal{T}_{target} \\
incomplete & \text{otherwise}
\end{cases}$$

</v-clicks>

</div>

</div>

---
theme: academic
---

# System Architecture

<div class="flex justify-center items-center h-4/5">
  <img src="/paper/draft/Architecture Figure.jpg" class="max-h-full w-auto" />
</div>

---
theme: academic
layout: two-cols
---

# Validation Domain

<div style="font-size: 11px;">

## Restaurant Guest Ordering

<v-clicks>

**Why restaurant ordering?**

✅ **Task Complexity**
- Multi-turn conversations
- Menu navigation
- Customization handling

✅ **State Management**
- Multiple items with modifiers
- Order building process
- Clarification handling

✅ **Behavioral Diversity**
- Different foodie personas
- Ordering styles
- Emotional responses

</v-clicks>

</div>

::right::

<div class="pl-4" style="font-size: 11px;">

<v-clicks>

## Dataset

📊 **20** diverse guest personas

🍽️ **50+** menu items with customizations

🧪 **60** test cases (3 per persona)

### Implementation
- **Framework**: Pydantic AI with GPT-4o
- **Tools**: Type-safe definitions
- **Logging**: Comprehensive tracking

### Ordering System
- LLM-based (GPT-4o) restaurant interface
- Natural language processing
- Menu knowledge & clarification handling
- Independent of guest simulation

</v-clicks>

</div>

---
theme: academic
---

# Evaluation Metrics (1/2)

<div class="grid grid-cols-2 gap-2" style="font-size: 10px;">

<div>

<v-click>

### Persona Adherence Score (PAS)

$$PAS = \frac{1}{N} \sum_{i=1}^{N} MS_i$$

$$MS_i = \sum_{j=1}^{4} w_j \cdot C_j$$

- $N$ = number of messages
- Equal weights $w_j = 0.25$ for:
  - Exploration style, mood tone
  - Task execution, completion status

</v-click>

<v-click>

### Task Restriction Adherence (TRA)

$$TRA = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$

$$Precision = \frac{|C \cap T|}{|C|}, Recall = \frac{|C \cap T|}{|T|}$$

- $T$ = normalized target items
- $C$ = normalized current state items

</v-click>

</div>

<div>

<v-click>

### Behavioral Variance Score (BVS)

$$TR_d = \frac{1}{M-1} \sum_{i=2}^{M} \mathbb{I}(state_i^d \neq state_{i-1}^d)$$

$$TR_{avg} = \frac{TR_{task\_execution\_style} + TR_{exploration\_style} + TR_{mood\_tone}}{3}$$

$$BVS = \begin{cases}
\frac{TR_{avg}}{0.2} & \text{if } TR_{avg} \leq 0.2 \\
1 - \frac{TR_{avg} - 0.2}{0.8} & \text{if } TR_{avg} > 0.2
\end{cases}$$

- $TR_d$ = transition rate for behavioral dimension $d$
- Optimal at 20% transition rate

</v-click>

<v-click>

### Decision Explainability Index (DEI)

$$DEI = \begin{cases}
0 & \text{No tools} \\
\min(0.2, \frac{ED}{N} \times 0.2) & \text{Basic tools} \\
\min(0.5, \frac{ED}{N} \times 0.5) & \text{Basic + 1 agent} \\
\min(1.0, \frac{ED}{2N}) & \text{Full system}
\end{cases}$$

- $ED$ = explained decisions, $N$ = messages

</v-click>

</div>

</div>

---
theme: academic
---

# Evaluation Metrics (2/2) - Composite Score

<div class="grid grid-cols-2 gap-2" style="font-size: 10px;">

<div>

<v-click>

### Composite Realism & Reliability Score (CRRS)

$$CRRS = 0.25 \cdot PAS + 0.20 \cdot BVS$$
$$+ 0.35 \cdot TRA + 0.20 \cdot DEI$$

**Unified score for overall simulation quality**

</v-click>

<v-click>

### Weight Distribution

- **TRA (35%)**: Primary task completion metric
- **PAS (25%)**: Persona consistency 
- **BVS (20%)**: Natural behavioral variation
- **DEI (20%)**: System validation & explainability

Task completion gets highest weight as it's critical for simulation success

</v-click>

</div>

<div>

<v-click>

### Metric Ranges & Interpretation

All metrics normalized to [0, 1] range:

- **PAS**: 1 = perfect persona adherence
- **BVS**: 1 = optimal variance (20% transitions)
- **TRA**: 1 = perfect F1 score for task items
- **DEI**: 1 = full explainability with tools
- **CRRS**: 1 = perfect overall simulation

</v-click>

<v-click>

### Performance Targets

- **CRRS > 0.8**: Excellent simulation quality
- **CRRS 0.6-0.8**: Good simulation quality
- **CRRS < 0.6**: Needs improvement

</v-click>

</div>

</div>

---

# Ablation Study Design

<div class="mt-8">

| Configuration | Order Tracking | Message Attributes | Architecture |
|---|:---:|:---:|---|
| **Config 1** (Baseline) | ❌ | ❌ | Single LLM |
| **Config 2** (User Only) | ❌ | ❌ | Single Agent |
| **Config 3** (User + ST) | ✅ | ❌ | Two Agents |
| **Config 4** (User + MAG) | ❌ | ✅ | Two Agents |
| **Config 5** (Full System) | ✅ | ✅ | Three Agents |

</div>

<v-clicks>

<div class="mt-8 text-center">
  <div class="text-2xl font-bold text-gradient">
    Testing contribution of each component systematically
  </div>
</div>

</v-clicks>

---
theme: academic
layout: center
---

# Results: Performance & Statistical Significance

<div class="grid grid-cols-2 gap-4" style="font-size: 11px;">

<div>

### Performance Metrics

| Config | PAS | BVS | TRA | DEI | CRRS |
|---|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.589 | 0.218 | 0.608 | 0.000 | 0.404 |
| **2** | 0.585 | 0.485 | 0.582 | 0.200 | 0.487 |
| **3** | 0.554 | 0.689 | **0.785** | 0.498 | 0.651 |
| **4** | **0.661** | 0.000 | 0.602 | 0.432 | 0.462 |
| **5** | 0.706 | **0.839** | **0.785** | **0.994** | **0.818** |

<v-click>

<div class="mt-4 text-center p-3 bg-gradient-to-r from-green-400/20 to-blue-500/20 rounded">
  <div class="text-base font-bold">
    🎯 Full Multi-Agent System Achieves 102.6% Improvement
  </div>
  <div class="text-xs mt-1">
    CRRS score doubles from 0.404 (baseline) to 0.818 (full system)
  </div>
</div>

</v-click>

</div>

<div style="font-size: 11px;">

### Statistical Significance
<div style="font-size: 10px;">(Baseline vs Full System)</div>

| Metric | Δ | p-value |
|---|---:|---|
| PAS | +20% | 0.004** |
| BVS | +285% | <0.001*** |
| TRA | +29% | 0.005** |
| DEI | +100% | <0.001*** |
| CRRS | +103% | <0.001*** |

<div style="font-size: 9px; margin-top: 2px;">** p < 0.01, *** p < 0.001</div>

<v-click>

<div class="mt-1 flex justify-center">
  <img src="/paper/draft/improvement_heatmap.png" class="h-36 w-auto rounded shadow" />
</div>

<div class="mt-1 text-center">
  <div style="font-size: 9px; font-weight: bold;">
    ✅ All metrics significant (p < 0.01)
  </div>
</div>

</v-click>

</div>

</div>

---

# Key Findings

<div class="grid grid-cols-2 gap-4 mt-4" style="font-size: 11px;">

<div>

### 🔗 Component Synergy

<v-clicks>

- Config 3 (State only): High TRA, low PAS
- Config 4 (Attrs only): High PAS, zero BVS
- **Config 5 (Full): Best across all metrics**

> Neither sub-agent alone achieves optimal performance

</v-clicks>

</div>

<div>

### 🤖 Behavioral Rigidity

<v-clicks>

- Pure behavioral control without state awareness
- Results in robotic, templated interactions
- **BVS = 0 for Config 4**

> State awareness is critical for natural variance

</v-clicks>

</div>

</div>

<v-click>

<div class="mt-4 p-3 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-lg" style="font-size: 11px;">

### 💰 Cost-Performance Trade-off

| Config | Avg Tokens | Latency (s) | CRRS |
|---|---:|---:|---:|
| Baseline | 6,618 | 5.08 | 0.404 |
| Full System | 14,789 | 23.16 | **0.818** |

**2.2x tokens for 2x performance improvement**

</div>

</v-click>

---
theme: academic
---

# Limitations, Ethics, Future Work & Applications

<div class="grid grid-cols-3 gap-4" style="font-size: 12px;">

<div>

<v-click>

### Current Limitations

- 📈 High computational cost (124% more tokens)
- 🔧 Domain-specific engineering required
- 🌐 English-only validation
- 🧠 Lacks complex behaviors

</v-click>

<v-click>

### Ethical Considerations

- 🔍 Maintain transparency
- ⚖️ Avoid bias perpetuation
- 🚫 No impersonation without consent
- 🛡️ Responsible deployment

</v-click>

</div>

<div>

<v-click>

### Future Directions

<div class="bg-blue-500/20 p-2 rounded mb-2">
  <div class="font-bold text-sm mb-1">🔄 Adaptive Evolution</div>
  <div style="font-size: 10px;">Dynamic behavioral adjustment</div>
</div>

<div class="bg-green-500/20 p-2 rounded mb-2">
  <div class="font-bold text-sm mb-1">🎭 Multi-Modal</div>
  <div style="font-size: 10px;">Voice, gesture, emotional tracking</div>
</div>

<div class="bg-purple-500/20 p-2 rounded mb-2">
  <div class="font-bold text-sm mb-1">🌍 Cross-Domain</div>
  <div style="font-size: 10px;">Healthcare, education, travel</div>
</div>

<div class="bg-orange-500/20 p-2 rounded">
  <div class="font-bold text-sm mb-1">⚡ Optimization</div>
  <div style="font-size: 10px;">Caching, selective invocation</div>
</div>

</v-click>

</div>

<div>

<v-click>

### Applications

**Testing & QA**
- Automated conversation testing
- Edge case discovery
- Performance benchmarking

**Broader Domains**
- 🗣️ Voice assistants
- 💬 Chatbots & virtual agents
- 🏥 Healthcare interfaces
- 🛍️ E-commerce platforms
- 📚 Educational tools
- 💼 Business automation

</v-click>

</div>

</div>

---

# References (1/2)

<div class="text-xs" style="column-count: 2; column-gap: 20px;">

1. Ahmad et al. (2025) "Simulating User Diversity"
2. Balog & Zhai (2025) "User Simulation in the Era of Generative AI"
3. Bernard & Balog (2024) "Formal Characterization of User Simulation"
4. Castricato et al. (2024) "PERSONA: Reproducible Testbed"
5. Cheng et al. (2024) "AutoPal: Autonomous Adaptation"
6. Chu et al. (2024) "Cohesive Conversations"
7. Chu et al. (2024) "Multimodal Emotional Support"
8. Dang et al. (2025) "Multi-Agent Collaboration"
9. Davidson et al. (2023) "User Simulation with LLMs"
10. Devanathan et al. (2025) "Why Synthetic Isn't Real Yet"
11. Feng et al. (2025) "Emotionally Intelligent Task-oriented Dialogue"
12. Ge et al. (2024) "PersonaHub: 1B Personas"
13. Hu & Ying (2025) "Unified Mind Model"
14. Hurst et al. (2024) "GPT-4o System Card"
15. Jia et al. (2024) "Leveraging LLMs for Dialogue Quality"
16. Lee et al. (2024) "OrchestraLLM: Efficient Orchestration"
17. Levi & Kadar (2025) "IntellAgent Framework"
18. Li et al. (2025) "LLM Generated Persona"
19. Liu et al. (2023) "AgentBench: Evaluating LLMs"
20. Maity & Deroy (2024) "Generative AI in Tutoring"
21. Mehri et al. (2025) "Goal Alignment in User Simulators"
22. Mo et al. (2024) "HierTOD: Hierarchical Goals"
23. Mohammadi et al. (2025) "Evaluation and Benchmarking"
24. Molchanova et al. (2025) "LLMs to Simulate Personality"
25. Niu et al. (2024) "Enhancing DST Models"
26. Park et al. (2023) "Generative Agents"
27. Park et al. (2024) "Simulations of 1,000 People"

</div>

---

# References (2/2)

<div class="text-xs" style="column-count: 2; column-gap: 20px;">

28. Park et al. (2024) "Generative Agent Simulations"
29. Park et al. (2025) "Simulating Human Behavior with AI"
30. Phy et al. (2020) "USLH Composite Metric"
31. PydanticAI (2024) "Python Agent Framework"
32. PydanticAI Docs (2024) "Documentation"
33. Rastogi et al. (2020) "Schema-Guided Dialogue"
34. Raza et al. (2024) "TRiSM for Agentic AI"
35. Saggar et al. (2025) "Score Before You Speak"
36. Shu et al. (2024) "Effective Multi-Agent Collaboration"
37. Sumers et al. (2024) "CoALA: Cognitive Architectures"
38. Sun et al. (2022) "Metaphorical User Simulators"
39. Suresh et al. (2025) "DiaSynth Framework"
40. Sutcliffe (2023) "Survey of Personality in Chatbots"
41. Tran et al. (2025) "Multi-Agent Collaboration Mechanisms"
42. Wakaki et al. (2024) "ComperDial Benchmark"
43. Wang & Chiu (2023) "Humanoid Agents Platform"
44. Wang et al. (2020) "KddRES Restaurant Dataset"
45. Wang et al. (2025) "Survey on LLM-based Agents"
46. Xiang et al. (2024) "Transformer Models for E-commerce"
47. Xie et al. (2025) "Human Simulacra Benchmark"
48. Xu et al. (2024) "Chain of Thought for DST"
49. Yi et al. (2024) "Multi-turn Dialogue Survey"
50. Yu et al. (2024) "AI Patient Simulation"
51. Zhang et al. (2025) "AgentOrchestra Framework"
52. Zhu et al. (2025) "Benchmarks & Evolutionary Evaluation"
53. Zhuge et al. (2024) "Agent-as-a-Judge"

</div>

<div class="mt-6 p-4 bg-blue-500/10 rounded-lg">
  <div class="font-bold mb-2">Code & Data Availability</div>
  <div class="text-sm">
    Implementation code, test datasets, and evaluation scripts available at:<br/>
    <code>https://github.toasttab.com/hareeshkarthik-toast/agentic-human-guest-simulation-for-ordering</code>
  </div>
</div>