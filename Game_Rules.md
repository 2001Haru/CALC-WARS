You could see the English version below the Chinese vesion.
# CALC WARS游戏规则说明文件
## 游戏模式
#### 在初始界面，您可以选择游玩经典模式或AI对战模式。其中，经典模式需要两名玩家线下游玩，分别为player1与player2，而AI对战模式则只需要一名玩家为player1与AI对战。每种模式中，您都可以选择对战方式(单局或BO5)，其中BO5是五局三胜制游戏并且每局交换先手权。在AI对战模式，您可以选择对战的AI难度，我们推荐您根据总游玩时长选择。与最高难度AI对战非常有趣，如果您拥有稳定击败其能力，说明您已经是此游戏的高手了！

## 流程介绍
#### 一局游戏需要两名玩家的参与。每局伊始，每名玩家拥有120HP与随机的5张数字牌与2张符号牌。所有玩家牌库互相可见且玩家需要使用这些卡牌进行计算特定的结果，以获得不同的效果或对对手造成不同的伤害。每一局拥有很多轮次(Rounds)，每一轮内又有许多回合(Turns)，玩家可以在每回合内多次操作。一方获胜的条件是：将对方HP清零。

### 牌库
#### 卡牌分为数字牌，符号牌与技能牌。数字牌含有1到13的自然数，每张牌上仅仅一个数字；符号牌是+，-，*，/，（，）的六种符号，每张牌上仅仅一个符号。技能牌则有诸多不同的类型，具体如下：
Factorial: One Heal Card( Recover 20 HP and draw 1 crad),
Cube: One Steal Card( Steal 2 cards randomly ),
Square: One Draw Card( Draw 2 numnber cards and 1 operator cards ),
24 Points: Two Shield Cards( Get 1 shield ),
ZERO: One Ruin card( Destroy 3 cards ramdomly),
ONE: One Pierce card( Break the opponent's shield )
注意，若双方手牌总数差距大于6张，优势方STEAL将只允许偷盗1张牌。
#### 如上左侧分别是指玩家计算得到的数字，右侧则是玩家计算这些特定数字得到的技能牌与其效果。如玩家1使用数字牌10，符号牌+，数字牌6，计算得到16，则可以获得Steal卡一张。此处特别说明，24算作阶乘与24点触发数字，0和1仅仅分别作为ZERO与ONE触发数字( 出于平衡性考虑不将0与1作为平方或立方或阶乘 )。某些技能卡牌效果会在后文继续具体介绍。

### 靶子，血量与护盾
#### 每一局都会生成一个固定的靶子，分为红黄蓝三个区域，其中：红色区域仅有一个数字，是36到56质数中的随机一个；黄色区域有两个数字，是24到36中的随机不同数字；蓝色区域有四个数字，是0到24中的随机不同数字，其中必定有一个平方数。如果玩家计算得到红色靶子中数字，且对手没有护盾时，对手减少50HP；黄色则减少30HP；蓝色则减少10HP。
#### 护盾是一种状态，玩家可以持有任意自然数数量的护盾，每张护盾可以抵挡任意数值伤害后破碎，而发动Pierce牌后可以将对手的护盾直接清零。

### 计算，轮次与先手权
#### 每局开始时，两名玩家进入第一轮，随机一名玩家拥有先手操作权，每名玩家被随机给予5张数字牌与2张符号牌。场上只能有一名玩家拥有操作权，拥有操作权时允许发动任意技能卡牌效果或进行任意有效计算。每次计算完成后，点击‘确认(Confirm)’即可得到结果，同时计算使用的卡牌会消失。需要注意的是，计算结果并不会作为新的数字牌加入牌库，否则括号牌失去意义；若点击‘结束(End the Turn)’，则将操作权交给对手,同时会被奖励任意2张牌；若点击‘结束本轮(End te Round)’，则本轮内该玩家无法进行任何操作，操作权一直属于对手，直至进入下一轮。进入下一轮后，上一轮先手点击‘结束本轮(End te Round)’的玩家拥有先手操作权，而后手操作玩家会被补偿2张牌，同时每名玩家再被给予随机5张牌，每两轮会增加一张给予牌数。

## 游戏平衡性
#### 我们经过长时间测试游玩后，对游戏已经做出了一些特别调整：首先，我们将0与1不作为阶乘或立方或平方数触发技能效果。初始时我们本不是如此设置，然而在实际游玩中，我们发现反复计算0与1的收益非常高，使得游戏完全失去了玩法的意义，这是我们不愿意看到的；其次，Pierce牌的设置是我们刻意调整加入的，原因是一直计算24点在原游戏中非常容易形成‘不败金身’，致使双方玩家谁也无法获取胜利。我们为1这个特殊的数字设计了Pierce牌，来打破这种僵局(实际上，1这个数字也很像一个矛头)；最后，关于Draw牌，Steal牌与Ruin牌的设计理念是，我们不会创造在任意条件下都有正收益的技能牌，所以在你对对手牌库进行毁坏或偷走操作时，自己也要付出使用卡牌的代价；同样的，抓取新的卡牌也是如此。

## 设计原则
#### 我们认为，一款优秀的策略游戏，需要秉持一条最重要的原则：所有局部最优操作的组合不等于全局游戏的最优操作。这意味着任何一步都需要玩家对未来操作的深思熟虑，而不是仅仅考虑当下收益。实际上，坚持这一条原则会让游戏变得更好玩。


# CALC WARS Game Rules

## Game Modes
#### At the start screen, you can choose between **Classic Mode** or **AI Battle Mode**. 
* **Classic Mode**: Requires two players to play locally (Player 1 vs. Player 2).
* **AI Battle Mode**: A single player (Player 1) competes against the AI. 
In either mode, you can select the match format: **Single Match** or **BO5 (Best of Five)**. BO5 is a "first to three" format where the initiative (starting turn) alternates each round. In AI Battle Mode, you can choose the AI difficulty; we recommend selecting a difficulty based on your total playtime. Playing against the highest-level AI is challenging—if you can defeat it consistently, you are a master of the game!

## Gameplay Flow
#### A match consists of two players. At the start, each player has **120 HP** and is randomly dealt **5 Number Cards** and **2 Operator Cards**. 
All player hands are visible to each other. Players use these cards to calculate specific results to trigger various effects or deal damage to the opponent. A match consists of multiple **Rounds**, and each round consists of multiple **Turns**. Players can perform multiple operations within a single turn. 
**Winning Condition:** Reduce the opponent's HP to zero.

### The Card Deck
#### Cards are categorized into Number Cards, Operator Cards, and Skill Cards.
* **Number Cards**: Contain natural numbers from 1 to 13.
* **Operator Cards**: The six symbols: `+`, `-`, `*`, `/`, `(`, `)`.
* **Skill Cards**: Triggered by calculating specific numerical results. The types are as follows:

| Calculation Result | Skill Card Type | Effect |
| :--- | :--- | :--- |
| **Factorial** (e.g., 24) | Heal Card | Recover 20 HP and draw 1 card |
| **Cube** (e.g., 8, 27) | Steal Card | Steal 2 cards randomly from the opponent |
| **Square** (e.g., 4, 9, 16) | Draw Card | Draw 2 number cards and 1 operator card |
| **24 Points** (24) | Shield Card | Receive 2 Shield cards (Gain 1 Shield per card) |
| **ZERO** (0) | Ruin Card | Randomly destroy 3 of the opponent's cards |
| **ONE** (1) | Pierce Card | Break all of the opponent's shields immediately |

*Note: If the hand size difference between players is greater than 6, a "Steal" action by the advantaged player is restricted to 1 card only.*
*Note on Balance: The number 24 triggers both Factorial and 24 Points. For balance reasons, the numbers 0 and 1 only trigger ZERO and ONE respectively; they do not trigger Square, Cube, or Factorial effects.*

### Targets, HP, and Shields
#### Every match generates fixed "Targets" divided into three zones:
* **Red Zone**: Contains one random prime number between 36 and 56. Deals **50 Damage**.
* **Yellow Zone**: Contains two different random numbers between 24 and 36. Deals **30 Damage**.
* **Blue Zone**: Contains four different random numbers between 0 and 24 (one is guaranteed to be a square number). Deals **10 Damage**.

**Shields:** A shield is a status. Players can hold any number of shields. One shield absorbs the total damage of a single attack before breaking. A **Pierce** card clears all opponent shields regardless of how many they have.

### Calculation, Rounds, and Initiative
#### Each round begins with a random player holding the initiative.
Only the player with the current "Turn" can perform calculations or use skill cards. 
* **Confirm**: After a calculation, click "Confirm" to apply the result. The used cards will be consumed. Calculated results do not become new cards (otherwise, brackets would lose their strategic purpose).
* **End the Turn**: Pass the turn to the opponent. As a reward, you will receive **2 random cards**.
* **End the Round**: The player can no longer act for the remainder of the round. The opponent retains the turn until they also choose to end the round. 
* **Next Round**: The player who first clicked "End the Round" in the previous round gains initiative for the new round. The player who acted last is compensated with **2 cards**. Additionally, both players are dealt **5 random cards** at the start of every round. The number of cards dealt increases by 1 every two rounds.

## Game Balance
#### After extensive playtesting, we have implemented specific adjustments:
1.  **0 and 1 Restrictions**: These numbers do not trigger Square, Cube, or Factorial effects. We found that the high efficiency of spamming 0 and 1 calculations trivialized the gameplay.
2.  **The Pierce Card**: This was added to prevent "Stalemates." In early builds, players could spam 24-point calculations to become invincible with shields. The "1" result (which visually resembles a spearhead) was designed as the Pierce card to break this deadlock.
3.  **Skill Costs**: Cards like Draw, Steal, and Ruin are designed so they are not "strictly positive" in all scenarios. You must weigh the cost of the cards used in the calculation against the benefit of the skill gained.

## Design Philosophy
#### We believe a great strategy game must follow one core principle: **The sum of local optimal operations does not equal the global optimal strategy.** This means every move requires the player to think deeply about future turns rather than just immediate gains. Adhering to this principle makes the game significantly more engaging and rewarding.