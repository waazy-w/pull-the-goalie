# Pull the Goalie Sooner

**Tagline:** NHL coaches empty the net about 1:40 too late when down one goal, and five minutes too late when down two. We built the model that shows the exact moment to pull, and audited every decision in 4,000 games.

**Live app:** https://waazy-w.github.io/pull-the-goalie/ (runs in the browser, first load about 30 s)  
**Code:** https://github.com/waazy-w/pull-the-goalie

---

## Inspiration
Pulling the goalie is the one high-stakes call a coach can make on a stopwatch. Analysts have argued for years that coaches wait too long. But nobody hands a bench boss a number they can trust for *their* situation, or shows them how much the habit costs. We wanted to turn "you should pull earlier" into "pull at 3:35, and here's what waiting cost you last season."

## What it does
**Pull the Goalie Sooner** is an interactive web app built on a decision model fit to **4,072 NHL regular-season games** (2023-24 through the 2026-27 season to date). It has four views:

* **The Verdict.** The optimal pull time vs. what the league actually does, and the comeback chance at every possible pull time.
* **Bench Calculator.** Enter the deficit, time left, manpower (even, power play, shorthanded, with the penalty clock) and team strength. You get a plain **PULL NOW / WAIT until m:ss** answer, plus a curve of the comeback chance by when you pull. One click loads any team's shrunk 5v5 profile.
* **Team Audit.** Every team-season ranked by **expected standings points lost** to late pulls, with each team's own optimal times next to its actual median pull times.
* **Game Replay.** Pick any late-game deficit: see the score margin, the empty-net window, goals, the model's pull moment vs. the actual one, and the comeback chance under each.

## Key findings
| | Model (95% CI) | NHL median |
|---|---|---|
| Down 1, even strength | **3:35** left (3:01 to 4:13) | 1:57 |
| Down 2, even strength | **7:21** left (6:08 to 8:48) | 2:42 |

* **94%** of 3,836 late-deficit decisions were made late, and the habit has not moved across four seasons.
* Waiting costs the league about **13 standings points per season**, and up to 0.66 points for a single team. It's small per game, but it's free: it costs no salary-cap space and needs no trade.
* Each team should pull at a different time. Teams with strong 5v5 offence should pull as early as **4:56** down 1; weak ones around **2:40**.
* On a power play, keep the goalie in a little longer: a 5-on-4 already gives most of the extra-attacker benefit, and an empty net on top adds a lot of risk.

## How we built it
1. **Data.** All regular-season play-by-play from the public NHL API (`api-web.nhle.com`). Each play's `situationCode` encodes whether each goalie is in net, so we recover every empty-net second. We separate **strategic pulls** from **delayed-penalty** extra attackers using the feed's `delayed-penalty` events, with a fallback rule for gaps in the feed.
2. **Scoring rates.** In the last 10 minutes of the third period, goals for and against the trailing team are modeled as Poisson processes for each state (goalie in or pulled × even / power play / shorthanded). Sparse cells are shrunk with a gamma prior.
3. **Decision model.** A finite-horizon **Markov decision process** solved by backward induction on a one-second grid over (time left, goal margin, power-play clock). The reward is **NHL standings points** (an OT loss is worth a point), not win probability. When the team is leading, the opponent follows the league's observed pull habits.
4. **Uncertainty and validation.** 200 game-level bootstrap resamples give confidence intervals on the optimal times. The rates were fit on 2023-25 only and then tested on 942 later real pulls. On those, the Brier score (lower is better) was 0.079, against 0.085 for a baseline that predicts the same comeback rate every time.
5. **Audit.** For every stint trailing by 1 or 2 that reaches the optimal moment, we compute expected points under the optimal policy minus expected points under the team's actual pull time. Stints where a goal came first are imputed with that team's typical pull time.
6. **App.** Python, pandas, NumPy, Plotly, Streamlit. The full pipeline is reproducible with two commands, and there are unit tests for the parser and the decision model.

## Actionable impact
* **For coaches:** a one-page bench card (down 1: 3:35, down 2: 7:21, adjusted for power plays and penalty kills) and a live calculator for any game state.
* **For front offices and analytics staff:** a team-level audit that quantifies the habit in standings points, the currency GMs are judged on, plus team-specific thresholds that account for roster strength.
* **For the league and media:** a transparent, open-source benchmark showing that this habit has not changed in four seasons.

## Challenges
* Telling a strategic pull from a delayed-penalty extra attacker, in a feed that only shows an empty net at the *next* recorded play.
* Picking the right objective. Standings points, not win probability, change the math, because reaching overtime is worth a point.
* Being honest about effect size. A single late pull costs about 0.016 points down 1. The story is that it happens about 1,000 times a season.

## What we learned
The math on pulling earlier has been around for years, and it has barely changed behavior. A rule only helps if a coach can use it on the bench, which is why we built the calculator and the cards.

## What's next
Add zone and possession at the moment of the pull (coaches pull with the puck in the offensive zone, so the model under-predicts comebacks in its top band). Add player-level on-ice strength, and a live mode that ingests tonight's games.

## Built with
python · pandas · numpy · plotly · streamlit · nhl-api · dynamic-programming · markov-decision-process · bootstrap
