%% HOW_TO_INTEGRATE.md — istruzioni per inserire le nuove sezioni

# Come integrare le nuove sezioni nel paper

## File prodotti

| File                          | Contenuto                                      | Da inserire dopo          |
|-------------------------------|------------------------------------------------|---------------------------|
| 02a_extended_phase_space.tex  | Extended phase space, Sundman, separabilità    | Inizio sezione 2 (prima di Yoshida) |
| 02b_symplecticity_proof.tex   | Dimostrazione formale simpletticità DKD        | Dopo 02a                  |
| 02c_time_reversal_proof.tex   | Dimostrazione reversibilità temporale          | Dopo 02b                  |
| 02d_order_and_shadow.tex      | Analisi BCH, ordine 4, shadow Hamiltonian      | Dopo 02c (sostituisce §2.5) |
| 02e_hessian_step_function.tex | Ottimalità Hessiana, Hessiana full model       | Dopo §2.3 esistente       |
| 03_stm_extended.tex           | STM con derivazione e dim. simpletticità       | Sostituisce sections/03_stm.tex |

## Modifiche a main.tex

Nella sezione 2, cambia la struttura così:

```latex
\section{The \AAS\ Integrator}

% Sezioni esistenti (2.1 Hamiltonian, 2.2 Yoshida) — invariate

\input{sections/02a_extended_phase_space}   % NUOVO
\input{sections/02b_symplecticity_proof}    % NUOVO
\input{sections/02c_time_reversal_proof}    % NUOVO

% Sezione 2.3 esistente (Hessian step size) — invariata
\input{sections/02e_hessian_step_function}  % NUOVO — aggiunge §§ a 2.3

\input{sections/02d_order_and_shadow}       % NUOVO — sostituisce §2.5
```

Per la sezione 3:
```latex
\input{sections/03_stm_extended}   % sostituisce sections/03_stm.tex
```

## Nuove macro necessarie in sections/macros.tex

Aggiungi:
```latex
% Per le dimostrazioni
\newcommand{\vq}{\mathbf{q}}   % già presente come \vq
\newcommand{\vp}{\mathbf{p}}   % già presente come \vp

% Operatori matematici
\DeclareMathOperator{\tr}{tr}
\newcommand{\Poisson}[2]{\left\{#1,#2\right\}}  % parentesi di Poisson

% Ambienti theorem (richiede \usepackage{amsthm} in main.tex)
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proof_env}{Proof}
```

In main.tex aggiungi nel preamble:
```latex
\usepackage{amsthm}
```

Nota: aastex631 non fornisce gli ambienti theorem/lemma —
amsthm è necessario.

## Stima lunghezza aggiuntiva

| Sezione                      | Righe LaTeX | Pagine PDF stimate |
|------------------------------|-------------|-------------------|
| 02a_extended_phase_space     | ~80         | ~1.2              |
| 02b_symplecticity_proof      | ~100        | ~1.5              |
| 02c_time_reversal_proof      | ~90         | ~1.3              |
| 02d_order_and_shadow         | ~80         | ~1.2              |
| 02e_hessian_step_function    | ~100        | ~1.5              |
| 03_stm_extended (espansione) | ~120        | ~1.8              |
| **Totale aggiunto**          | **~570**    | **~8.5 pagine**   |

Il paper passerà da ~14 a ~22-23 pagine — appropriato per MNRAS.

## Dipendenze tra sezioni

Le sezioni usano queste etichette che devono essere presenti:

Dal file 02a:
  - \label{eq:H_separable}
  - \label{eq:extended_hamiltonian}
  - \label{eq:Gamma_split}
  - \label{eq:V_flow}
  - \label{eq:T_flow_ode}

Dal file 02b (richiede da 02a):
  - \label{sec:extended_phase_space}
  - \label{eq:drift_map}, \label{eq:kick_map}
  - \ref{eq:dkd} — deve essere definita in 02_method.tex

Dal file 02c (richiede da 02b):
  - \ref{eq:time_reversal_def}
  - \ref{lem:harmonic_symmetry}

Dal file 02d (richiede da 02b, 02c):
  - \ref{eq:g_function} — in 02_method.tex
  - \ref{sec:shadow} — questa sezione
  - \ref{fig:energy_vs_precision} — in 04_benchmarks.tex

Dal file 02e (richiede da 02d):
  - \ref{eq:A_matrix} — in 03_stm_extended.tex o 02_method.tex
  - \ref{sec:longterm} — in sezione 5

Dal file 03_stm_extended (richiede da 02b):
  - \ref{eq:kick_jacobian}, \ref{eq:drift_jacobian} — in 02b
  - \ref{sec:stm} — questa sezione

## Note per l'agente LaTeX

1. Rimuovi l'attuale \subsection{Shadow Hamiltonian} da 02_method.tex
   (ora coperta da 02d_order_and_shadow.tex)

2. Rimuovi la nota "proof sketch" se presente

3. Verifica che \newtheorem sia dichiarato una sola volta

4. Controlla che tutte le \ref{} siano risolte dopo la compilazione
