# 24-Points Card Game
## Origin of the Idea
#### We believe the gameplay of the classic '24 Points' game is very interesting: given any four natural numbers between 1 and 10, players try to use addition, subtraction, multiplication, division, and parentheses to calculate the result 24. However, the traditional 24 Points game lacks competitiveness and strategic depth, making it easy for players to get bored. Therefore, we aimed to develop a small game based on the 24 Points concept to fulfill our pure pursuit of 'fun' gameplay.

## Process Introduction
#### A single game requires two players. At the start, each player has 100 HP and is randomly dealt 7 cards. Players must use these cards to calculate specific results, which can trigger various effects or deal damage to the opponent. A game consists of multiple rounds, and within each round, players can take several actions. The winning condition is to reduce the opponent's HP to zero.

### Card Library
#### Cards are divided into Number Cards, Symbol Cards, and Skill Cards. Number Cards contain natural numbers from 1 to 13, with one number per card. Symbol Cards include the six symbols: +, -, *, /, (, ), with one symbol per card. Skill Cards have various types, detailed as follows:
##### Factorial: One Heal Card (Recover 30 HP and draw 1 card),
##### Cube: One Steal Card (Steal 3 cards randomly),
##### Square: One Draw Card (Draw 3 cards randomly),
##### 24 Points: Two Shield Cards (Get 1 shield),
##### ZERO: One Ruin card (Destroy 3 cards randomly),
##### ONE: One Pierce card (Break the opponent's shield)
#### The left side indicates the number calculated by the player, and the right side indicates the Skill Card obtained and its effect. For example, if Player 1 uses a Number Card 10, a Symbol Card +, and a Number Card 6 to calculate 16, they receive one Steal Card. It is specially noted that 24 is treated as both a factorial and the 24 Points trigger number, while 0 and 1 are only treated as ZERO and ONE trigger numbers respectively (for balance reasons, 0 and 1 are not considered squares, cubes, or factorials). The effects of some Skill Cards will be explained in more detail later.

### Target, HP, and Shield
#### Each game generates a fixed target with three zones: Red, Yellow, and Blue. The Red zone contains one number, a random prime between 36 and 56. The Yellow zone contains two numbers, two distinct random numbers between 24 and 36. The Blue zone contains four numbers, four distinct random numbers between 0 and 24, including two perfect squares. If a player calculates the number in the Red zone and the opponent has no shield, the opponent loses 50 HP; for the Yellow zone, 25 HP; for the Blue zone, 10 HP.
#### A Shield is a status. A player can hold any natural number of shields. Each shield can block damage of any value once before breaking. Using a Pierce card directly sets the opponent's shield count to zero.

### Calculation, Rounds, and Turn Priority
#### At the start of the game, both players enter the first round. One player is randomly given the priority to act first. Each player is randomly dealt 7 Number or Symbol cards. Only one player has the right to act at any time. During their turn, a player can activate any Skill Card effect or perform any valid calculation. After each calculation, clicking 'Confirm' finalizes the result, and the cards used in the calculation are discarded. Note: The calculation result does not become a new Number Card in the library, otherwise parenthesis cards would lose their meaning. Clicking 'End' passes the turn to the opponent and rewards the player with 2 random cards. Clicking 'End Round' means the player cannot take any more actions for the remainder of the round, and the turn permanently goes to the opponent until the next round begins. When a new round starts, the player who clicked 'End Round' first in the previous round gets the turn priority. The player who acted second in the previous round is compensated with 2 cards. Additionally, each player is dealt 5 random cards at the start of a new round, and this number increases by 1 card every two rounds.

## Game Balance
#### After extensive testing and gameplay, we have made specific adjustments: Firstly, 0 and 1 are not treated as factorials, cubes, or squares. Initially, they were, but we found that repeatedly calculating 0 or 1 yielded excessively high rewards, diverting the game from its core 24 Points gameplay, which was undesirable. Secondly, the Pierce card was intentionally added. Originally, consistently calculating 24 could easily create an 'invincible' state, leading to stalemates. We designed the Pierce card for the number 1 (which resembles a spearhead) to break such deadlocks. Finally, the design philosophy behind Draw, Steal, and Ruin cards is that no Skill Card should provide unconditional positive benefits. Thus, when destroying or stealing cards from the opponent's library, the player also expends resources (using cards). Similarly, drawing new cards also comes at the cost of an action.

##Design Principles
####We believe that a key principle for an excellent strategy game is: ​The combination of all locally optimal moves does not equal the globally optimal strategy for the entire game.​​ This means every move requires players to think deeply about future consequences, not just immediate gains. Adhering to this principle ultimately makes the game more engaging and fun.



# 二十四点卡牌游戏
## 想法起源
#### 我们认为经典游戏‘二十四点’的玩法非常有趣：任意给予四个1到10之间的自然数，尝试使用加减乘除与括号计算得出24。然而，传统二十四点玩法的竞技性与思维深度是绝对的短板，非常容易使人感到乏味。因此，我们希望开发一款基于二十四点的小游戏，来满足我们对‘好玩’的纯粹追求。

## 流程介绍
#### 一局游戏需要两名玩家的参与。每局伊始，每名玩家拥有100HP与随机的7张牌。玩家需要使用这些卡牌进行计算特定的结果，以获得不同的效果或对对手造成不同的伤害。每一局拥有很多轮次，每一轮内玩家可以多次操作。一方获胜的条件是：将对方HP清零。

### 牌库
#### 卡牌分为数字牌，符号牌与技能牌。数字牌含有1到13的自然数，每张牌上仅仅一个数字；符号牌是+，-，*，/，（，）的六种符号，每张牌上仅仅一个符号。技能牌则有诸多不同的类型，具体如下：
##### Factorial: One Heal Card( Recover 30 HP and draw 1 crad),
##### Cube: One Steal Card( Steal 3 cards randomly ),
##### Square: One Draw Card( Draw 3 cards randomly ),
##### 24 Points: Two Shield Cards( Get 1 shield ),
##### ZERO: One Ruin card( Destroy 3 cards ramdomly),
##### ONE: One Pierce card( Break the opponent's shield )
#### 如上左侧分别是指玩家计算得到的数字，右侧则是玩家计算这些特定数字得到的技能牌与其效果。如玩家1使用数字牌10，符号牌+，数字牌6，计算得到16，则可以获得Steal卡一张。此处特别说明，24算作阶乘与24点触发数字，0和1仅仅分别作为ZERO与ONE触发数字( 出于平衡性考虑不将0与1作为平方或立方或阶乘 )。某些技能卡牌效果会在后文继续具体介绍。

### 靶子，血量与护盾
#### 每一局都会生成一个固定的靶子，分为红黄蓝三个区域，其中：红色区域仅有一个数字，是36到56质数中的随机一个；黄色区域有两个数字，是24到36中的随机不同数字；蓝色区域有四个数字，是0到24中的随机不同数字，其中必定有两个平方数。如果玩家计算得到红色靶子中数字，且对手没有护盾时，对手减少50HP；黄色则减少25HP；蓝色则减少10HP。
#### 护盾是一种状态，玩家可以持有任意自然数数量的护盾，每张护盾可以抵挡任意数值伤害后破碎，而发动Pierce牌后可以将对手的护盾直接清零。

### 计算，轮次与先手权
#### 每局开始时，两名玩家进入第一轮，随机一名玩家拥有先手操作权，每名玩家被随机给予7张数字牌或符号牌。场上只能有一名玩家拥有操作权，拥有操作权时允许发动任意技能卡牌效果或进行任意有效计算。每次计算完成后，点击‘确认’即可得到结果，同时计算使用的卡牌会消失。需要注意的是，计算结果并不会作为新的数字牌加入牌库，否则括号牌失去意义；若点击‘结束’，则将操作权交给对手,同时会被奖励任意2张牌；若点击‘结束本轮’，则本轮内该玩家无法进行任何操作，操作权一直属于对手，直至进入下一轮。进入下一轮后，上一轮先手点击‘结束本轮’的玩家拥有先手操作权，而后手操作玩家会被补偿2张牌，同时每名玩家再被给予随机5张牌，每两轮会增加一张给予牌数。

## 游戏平衡性
#### 我们经过长时间测试游玩后，对游戏已经做出了一些特别调整：首先，我们将0与1不作为阶乘或立方或平方数。初始设计时我们本不是如此设置，然而在实际游玩中，我们发现反复计算0与1的收益非常高，使得游戏完全失去了原本以24点为核心玩法的意义，这是我们不愿意看到的；其次，Pierce牌的设置是我们刻意调整加入的，原因是一直计算24点在原游戏中非常容易形成‘不败金身’，致使双方玩家谁也无法获取胜利。我们为1这个特殊的数字设计了Pierce牌，来打破这种僵局( 实际上，1这个数字也很像一个矛头)；最后，关于Draw牌，Steal牌与Ruin牌的设计理念是，我们不会创造在任意条件下都有正收益的技能牌，所以在你对对手牌库进行毁坏或偷走操作时，自己也要付出使用卡牌的代价；同样的，抓取新的卡牌也是如此。

## 设计原则
#### 我们认为，一款优秀的策略游戏，需要秉持一条最重要的原则：所有局部最优操作的组合不等于全局游戏的最优操作。这意味着任何一步都需要玩家对未来操作的深思熟虑，而不是仅仅考虑当下收益。实际上，坚持这一条原则会让游戏变得更好玩。
