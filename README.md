# PPO-Agent für Gangverhalten in Humanoid-v5

## Projektübersicht

Dieses Projekt untersucht den Einsatz von Proximal Policy Optimization (PPO)
für das Erlernen von Gangverhalten in einer angepassten
`Humanoid-v5`-Umgebung auf Basis von Gymnasium und MuJoCo.

Im Mittelpunkt steht die Frage, inwiefern sich durch gezielte Anpassungen der
Reward-Funktion bestimmte Eigenschaften des Bewegungsverhaltens fördern lassen.
Dazu wurden unter anderem Belohnungs- und Strafkomponenten für symmetrisches
Verhalten, unerwünschte Sprungbewegungen und die Stabilisierung der
Körperhaltung untersucht.

Die entwickelte Umgebung und der PPO-Agent wurden anschließend anhand des
erlernten Bewegungsverhaltens und verschiedener Trainingsmetriken evaluiert.

## Zielsetzung

- Implementierung eines PPO-Agenten für die Humanoid-v5-Umgebung
- Untersuchung des Einflusses verschiedener Reward-Komponenten auf das
  Bewegungsverhalten
- Förderung eines symmetrischeren und stabileren Bewegungsmusters
- Quantitative Auswertung des Trainingsverlaufs
- Qualitative Bewertung des resultierenden Bewegungsverhaltens

## Ergebnis

Die Experimente zeigten, dass der Agent ein Bewegungsmuster erlernen konnte,
das grundsätzlich eine Vorwärtsbewegung des Humanoiden ermöglicht.

Das resultierende Verhalten entsprach jedoch nicht dem ursprünglich angestrebten
natürlichen menschlichen Gang. Statt eines ausgeprägten, alternierenden
Gehzyklus entwickelte die Policy überwiegend ein kurzes, tippelndes
Bewegungsmuster.

Damit konnte das ursprüngliche Ziel eines natürlichen und robusten
Gangverhaltens nicht vollständig erreicht werden.

Die Ergebnisse zeigen gleichzeitig die Herausforderungen beim Reward Design
für komplexe humanoide Bewegungsaufgaben. Insbesondere die Gewichtung der
verschiedenen Reward-Komponenten beeinflusst das erlernte Verhalten stark und
kann dazu führen, dass der Agent eine lokal vorteilhafte, aber nicht
menschlich anmutende Bewegungsstrategie entwickelt.

## Beispiel

![Beispiel GIF](./Kurzesvideo.gif)

[Video auf YouTube](https://youtube.com/shorts/yQRjeEFj1OA?feature=share)

## Technischer Ansatz

### Reinforcement Learning

Als Lernverfahren wird Proximal Policy Optimization (PPO) verwendet.

Der Agent besteht aus separaten neuronalen Netzen für:

- Policy
- Value Function

Die Policy bestimmt die Aktionen des Humanoiden, während das Value Network
den erwarteten zukünftigen Return schätzt.

### Angepasste Humanoid-Umgebung

Die originale Humanoid-v5-Umgebung wurde um zusätzliche Reward-Komponenten
erweitert.

Dabei wurden unter anderem folgende Aspekte untersucht:

- Symmetrie des Bewegungsverhaltens
- Bestrafung unerwünschter Sprung-/Hüpfbewegungen
- Stabilisierung der Körperhaltung
- Vorwärtsbewegung

Das Ziel war es, den Lernprozess gezielt in Richtung eines stabileren und
symmetrischeren Gangverhaltens zu beeinflussen.

## Projektstruktur

```text
env.py         → modifizierte Humanoid-Umgebung
                 (Symmetrie + Anti-Hopping + Stabilisierung)

ppo.py         → PPO-Agent
                 (Policy, Value Network und Update-Loop)

mlp.py         → MLP-Module für Policy und Value Network

utils.py       → Hilfsfunktionen
                 (RunningNorm, explained_variance, etc.)

main.py        → Trainingsskript und vollständiger Training Loop

videos/        → gerenderte MP4-Videos und GIFs

logs/          → Trainings-Logs (CSV + TensorBoard)

plots/         → während des Trainings erzeugte Plots

checkpoints/   → gespeicherte Model-Checkpoints
