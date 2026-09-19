# Routing tests

The 25 labeled requests in [`relay/src/horizon_relay/fixtures/routing_tests.json`](relay/src/horizon_relay/fixtures/routing_tests.json). Each one has the model expected to answer it and, where there is one, the correct answer.

Run them from Open WebUI with the **Horizon Relay Tests** model: send `/run all`, or a subset such as `/run 1 4 9`, `/run 0.9b`, `/run 4b`, `/run cloud` or `/run forced`. Each run first empties the **Tests** folder, then saves one chat per test there with its decision graph. From a terminal: `horizon-relay eval [selection]`.

**Expected model** is the label in the test set; a test passes its route check when one of the listed models answers it. **Auto-checked** means the runner also checks the answer against a pattern. The others need a person to read the answer.

## 0.9B: short, literal, easily checked

| # | Question | Expected model | Correct answer | Auto-checked |
|---|---|---|---|---|
| 1 | Extract only the city from this sentence: The conference will take place in Detroit, Michigan. | 0.9B | Detroit | ✓ |
| 2 | Classify the sentiment as positive, neutral, or negative: I absolutely loved this movie. | 0.9B | Positive | ✓ |
| 3 | What is 17 times 23? | 0.9B | 391 | ✓ |
| 4 | What is the capital of France? Answer with only the city. | 0.9B | Paris | ✓ |
| 5 | Rewrite this sentence in the past tense: The researcher analyzes the experiment. | 0.9B | The researcher analyzed the experiment. | ✓ |
| 6 | Extract the date and city from: The K2 Horizon hackathon takes place September 19, 2026 in Ann Arbor, Michigan. | 0.9B | September 19, 2026; Ann Arbor | ✓ |
| 7 | Classify this support request as billing, technical, or general: My account was charged twice for the same month. | 0.9B | Billing | ✓ |

## 4B: moderate reasoning over given information

| # | Question | Expected model | Correct answer | Auto-checked |
|---|---|---|---|---|
| 8 | Which number is larger: 0.73 or 0.703? | 0.9B or 4B | 0.73 | ✓ |
| 9 | Find the bug in this Python function and briefly explain the fix: `def total(x): s = 0; for i in range(len(x)+1): s += x[i]; return s` | 4B | `range(len(x)+1)` reads one element past the end (IndexError). Use `range(len(x))`, or simply `sum(x)`. | ✓ |
| 10 | A price increases by 20 percent and then decreases by 20 percent. Is the final price equal to the original price? Explain briefly. | 4B | No. 1.2 × 0.8 = 0.96, so the final price is 4% lower. | ✓ |
| 11 | A recipe needs 3/4 cup of sugar per batch. How many cups are needed for 7 batches, and how many tablespoons is that? (1 cup = 16 tablespoons) | 4B | 5.25 cups = 84 tablespoons | ✓ |
| 12 | A meeting starts at 3:00 PM in New York (EDT, UTC-4) on September 19. What time does it start in London (BST, UTC+1) and in Tokyo (JST, UTC+9)? Give the date too if it changes. | 4B | London 8:00 PM, September 19; Tokyo 4:00 AM, September 20 | ✓ |
| 13 | Write a SQL query that returns the names of the three customers with the highest total order amount, given tables customers(id, name) and orders(id, customer_id, amount). | 4B | `SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.id, c.name ORDER BY SUM(o.amount) DESC LIMIT 3;` | ✓ (checks for `GROUP BY`) |
| 14 | Debug this reasoning: If two events each have probability 1/2, then their intersection must have probability 1/4. Explain exactly what assumption is missing. | 4B | Independence. Without it, P(A∩B) can be anything from 0 to 1/2. | ✓ |
| 15 | A fair coin is repeatedly flipped until two consecutive heads occur. What is the expected number of flips? Derive the answer carefully. | 4B or 375B | 6 | ✓ |

## 375B (cloud): hard multi-step reasoning or broad planning

| # | Question | Expected model | Correct answer | Auto-checked |
|---|---|---|---|---|
| 16 | A family has two children. You are told that at least one child is a boy born on Tuesday. Assuming independent equally likely sexes and weekdays, what is the probability both children are boys? Explain your reasoning. | 375B | 13/27 | ✓ |
| 17 | Alice, Bob, Cara, and David must each give one presentation in slots 1 through 4. Alice must present before Bob. Cara cannot present first or last. David must present immediately after Alice. Determine every valid schedule and explain how you know the list is complete. | 375B | Exactly one schedule: Alice, David, Cara, Bob | ✓ |
| 18 | You have 12 visually identical balls. One has a different weight, but you do not know whether it is heavier or lighter. Using a balance scale exactly three times, give a complete strategy that always identifies the odd ball and whether it is heavier or lighter. | 375B | The standard strategy: weigh 4 against 4; depending on the result, weigh a mixed 3-and-3 group; one final comparison settles it. Any complete, correct three-weighing strategy counts. | – |
| 19 | Suppose all A are B, some B are C, no C are D, and every D is A. Determine which relationships among A, B, C, and D must be empty or nonempty, and justify each conclusion. | 375B | Must be empty: C∩D. Must be nonempty: B, C and B∩C. Also D ⊆ A ⊆ B. Undetermined: whether A, D and A∩C are empty. | – |
| 20 | Consider a Python algorithm with an outer loop running n times. On iteration i, an inner loop runs i times, and each inner iteration performs a binary search over n elements. Derive the asymptotic time complexity carefully. | 375B | Θ(n² log n) | ✓ |
| 21 | Alex and Blake repeatedly roll separate fair six-sided dice. Alex rolls first. The first player to roll a 6 wins, but after every complete round in which neither rolls a 6, Alex loses one future turn. Determine Alex's probability of eventually winning and explain your state model. | 375B | Ambiguous as written. If lost turns accumulate, Alex only ever rolls once, giving 1/6. Judge the reasoning and stated model. | – |
| 22 | Plan a two-day prototype for a campus lost-and-found app. Compare two storage choices, suggest a minimal feature set, and identify the biggest implementation risks. | 375B | Open-ended; there is no single answer. It exercises cloud planning and delegation. | – |

## Forced paths: the policy is overridden so a specific route always runs

| # | Question | Expected model | Correct answer | Auto-checked |
|---|---|---|---|---|
| 23 | What is 17 times 23? (the 0.9B's threshold is 0, so it is never trusted) | 4B | 391 | ✓ |
| 24 | Same as #22 (no local answer is trusted; the cloud plans, delegates subtasks back to the local models, and synthesizes) | 375B | Open-ended | – |
| 25 | Same as #16 (no local answer is trusted; the cloud answers in one call) | 375B | 13/27 | ✓ |
