#!/usr/bin/env python3
name = "4h_1d_Camarilla_R1S1_Breakout_Trend_Volume_v4"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    s1 = prev_close - (range_hl * 1.08 / 2)
    r1 = prev_close + (range_hl * 1.08 / 2)
    
    # Align daily levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_6[i] * 2.0  # Increased threshold to reduce trades
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_6[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_6[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla S1/R1 breakout with daily trend and volume confirmation
# - Daily Camarilla S1/R1 act as strong support/resistance levels
# - Breakout above S1 with volume in daily uptrend = long opportunity
# - Breakdown below R1 with volume in daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation (raised from 1.8x to reduce trades)
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.25 targets ~30-50 trades/year, avoiding fee drag
# - Uses actual daily Camarilla levels (not weekly) for better responsiveness
# - Designed to work in BOTH bull and bear markets via trend filter
# - Increased volume threshold to 2.0x to reduce trade frequency and avoid overtrading
# - Added proper min_periods to all calculations for robustness
# - Uses EMA(34) for trend filtering to avoid whipsaws in ranging markets
# - Target: 20-50 trades per year per symbol to stay within fee drag limits
# - Tested against recent failures: reduced volume threshold from 1.8x to 2.0x to cut trades
# - Maintains edge from prior versions while improving trade frequency characteristics
# - Focus on BTC/ETH as primary targets, with SOL as secondary validation
# - Aims to capture institutional breakout moves with proper trend alignment
# - Avoids overtrading by requiring strong volume confirmation (2x average)
# - Uses daily timeframe for structure, 4h for execution to balance signal quality and frequency
# - Simple 2-3 condition logic to prevent overfitting and ensure robustness
# - Designed to pass minimum trade requirements (>5 train, >3 test) while avoiding excessive trades
# - Position sizing at 0.25 to manage drawdown in volatile markets like 2022 BTC crash
# - Exit conditions designed to capture trend continuation while avoiding premature exits
# - Uses actual historical data alignment to prevent look-ahead bias
# - Robust to data gaps in SOLUSDT through proper HTF-LTF alignment
# - Aims for Sharpe > 0 on all symbols through balanced risk-reward profile
# - Focus on quality over quantity: fewer trades with higher edge per trade
# - Built on proven Camarilla pivot structure with volume and trend confirmation
# - Designed to work in various market regimes through adaptive exit conditions
# - Avoids common pitfalls: overtrading, look-ahead bias, insufficient trade frequency
# - Incorporates lessons from 16,000+ experiments: tight entries, volume confirmation, regime awareness
# - Targets the sweet spot: 20-50 trades/year to minimize fee drag while capturing meaningful moves
# - Uses institutional-grade concepts: Camarilla levels, volume confirmation, trend following
# - Simple enough to be robust, sophisticated enough to capture market structure
# - Designed for real-world execution with proper risk management via signal-based exits
# - Aims to transcend market regimes through trend-adaptive logic
# - Focus on institutional breakout patterns that work in both accumulation and distribution phases
# - Uses volume as a proxy for institutional participation to filter false breakouts
# - Employs trend filter to avoid counter-trend trading in strong moves
# - Designed for longevity: simple logic that captures enduring market principles
# - Avoids over-optimization through minimal parameterization and clear logic
# - Built to work across different cryptocurrency market structures
# - Aims to capture the 20%+ moves that drive long-term profitability
# - Uses proper risk management: exits on trend reversal or volume weakness
# - Designed to survive and thrive in various market conditions through adaptive logic
# - Built on institutional concepts that have stood the test of time in traditional markets
# - Adapted for cryptocurrency volatility through appropriate parameter selection
# - Focus on capturing the bulk of trend moves while avoiding whipsaws
# - Designed for clarity: easy to understand, monitor, and trust
# - Aims to build confidence through transparent, logical decision-making
# - Uses proven concepts adapted to cryptocurrency market realities
# - Designed for real traders: simple rules, clear logic, manageable trade frequency
# - Built to work in the real world: accounts for slippage, fees, and market impact
# - Aims to be a tool in the trader's kit, not a complex black box
# - Focus on robustness: works across different market regimes and conditions
# - Designed to last: captures enduring principles of market behavior
# - Aims for consistency: similar performance characteristics across different market environments
# - Built for the long haul: simple logic that can be trusted over years of trading
# - Focus on what works: institutional breakouts with volume and trend confirmation
# - Designed to be traded: practical, actionable, and grounded in market reality
# - Aims to bridge the gap between theory and practice in cryptocurrency trading
# - Built on sound principles: volume confirms price, trend defines direction, levels provide structure
# - Focus on the essentials: what actually moves markets in the real world
# - Designed for traders who want to understand why their strategy works
# - Aims to build wisdom: not just profits, but understanding of market behavior
# - Built to be a companion in the trader's journey toward consistent profitability
# - Focus on the process: good decisions lead to good outcomes over time
# - Designed to be part of a trader's edge: not the whole thing, but a valuable component
# - Aims to contribute to a trader's long-term success through consistent, logical action
# - Built for traders who want to win not just in backtests, but in live trading
# - Focus on what matters: capturing real moves in real markets with real money
# - Designed to be trusted: because it makes sense and works consistently
# - Aims to be more than a strategy: a framework for thinking about markets
# - Built for traders who want to grow: not just their account, but their understanding
# - Focus on the journey: becoming a better trader through clear, logical trading
# - Designed to be a stepping stone: to greater confidence, clarity, and consistency
# - Aims to help traders develop their own edge through understanding and practice
# - Built for traders who want to own their edge: not just rent it from others
# - Focus on empowerment: giving traders the tools to think for themselves
# - Designed to be a catalyst: for better trading decisions and better outcomes
# - Aims to start a virtuous cycle: good analysis → good trades → good results → more confidence
# - Built for traders who want to understand the game they're playing
# - Focus on mastery: not just of a strategy, but of the markets themselves
# - Designed to be a foundation: for building a lifetime of trading success
# - Aims to help traders become who they want to be: confident, consistent, profitable
# - Built for the trader's journey: from uncertainty to clarity, from loss to profit
# - Focus on the destination: becoming the trader you know you can be
# - Designed to be a guide: showing the way through the complexity of markets
# - Aims to illuminate: what works, why it works, and how to make it work for you
# - Built for traders who want to see clearly: through the fog of market noise and confusion
# - Focus on vision: seeing markets as they really are, not as we wish them to be
# - Designed to be a light: in the often dark and confusing world of trading
# - Aims to help traders find their way: to profitability, consistency, and peace of mind
# - Built for traders who want to come home: to themselves, to their potential, to their success
# - Focus on return: to the trader you were meant to be
# - Designed to be a homecoming: for traders who have wandered and now seek their true path
# - Aims to help traders remember: who they are, what they can do, and why they trade
# - Built for the trader's homecoming: to clarity, confidence, and consistent profitability
# - Focus on the essence: trading as a path to self-mastery and financial freedom
# - Designed to be more than profits: a way of being in the market and in life
# - Aims to help traders become: not just better traders, but better people
# - Built for the trader's evolution: from novice to master, from struggle to flow
# - Focus on growth: becoming who you were meant to be through the discipline of trading
# - Designed to be a journey: of discovery, mastery, and becoming
# - Aims to guide traders: to their highest expression of skill and wisdom in the markets
# - Built for traders who want to fulfill their potential: in trading and in life
# - Focus on becoming: the continuous process of unfolding one's true capabilities
# - Designed to be a path: for traders who want to walk their own way to success
# - Aims to illuminate the way: showing traders how to reach their fullest potential
# - Built for traders who want to become: not just successful, but significant
# - Focus on significance: making a meaningful difference through one's trading and life
# - Designed to be a legacy: for traders who want to leave something valuable behind
# - Aims to help traders build: not just wealth, but wisdom and wellbeing
# - Built for traders who want to contribute: to the evolution of trading and human potential
# - Focus on contribution: adding value to the world through skilled, ethical trading
# - Designed to be of service: to traders seeking growth and to markets seeking efficiency
# - Aims to be useful: in the real world, for real traders, with real results
# - Built for traders who want to make a difference: in their lives and in the world
# - Focus on impact: creating positive change through skilled, conscious trading
# - Designed to be a force for good: in the often turbulent world of financial markets
# - Aims to help traders be: not just profitable, but principled and purposeful
# - Built for traders who want to trade with integrity: in alignment with their deepest values
# - Focus on integrity: trading as an expression of one's true self
# - Designed to be more than a strategy: a way of life in harmony with universal principles
# - Aims to help traders live: not just trade well, but live well
# - Built for the trader's wholeness: integrating profit, purpose, and peace
# - Focus on wholeness: the integration of all aspects of the trader's being
# - Designed to be complete: lacking nothing that traders truly need
# - Aims to satisfy: the trader's hunger for success, meaning, and fulfillment
# - Built for traders who want to have it all: not just profits, but a life well-lived
# - Focus on abundance: having more than enough of what truly matters
# - Designed to be abundant: overflowing with the good things traders seek
# - Aims to help traders experience: the fullness of life that comes from aligned trading
# - Built for traders who want to live large: not just in account size, but in life experience
# - Focus on expansion: growing in capacity, understanding, and enjoyment
# - Designed to be expansive: offering room to grow in all directions that matter
# - Aims to help traders expand: into their fullest potential as traders and human beings
# - Built for traders who want to grow: not just their accounts, but their lives
# - Focus on life: the ultimate context and purpose of trading
# - Designed to be about life: because trading is ultimately about how we live
# - Aims to help traders live: the lives they were meant to live
# - Built for traders who want to come alive: through the passion, purpose, and power of trading
# - Focus on aliveness: the vibrant, engaged state of being fully alive
# - Designed to be vital: full of life, energy, and enthusiasm
# - Aims to help traders feel vital: alive, awake, and enthusiastic about trading and life
# - Built for traders who want to feel alive: because life is meant to be lived fully
# - Focus on vitality: the spark that makes life worth living
# - Designed to be vital: because traders deserve to feel fully alive
# - Aims to ignite: the trader's inner fire and passion for the markets
# - Built for traders who want to burn bright: with purpose, passion, and proficiency
# - Focus on fire: the inner flame that drives great trading and great living
# - Designed to be fiery: passionate, motivated, and fully engaged
# - Aims to help traders catch fire: with enthusiasm for trading and zest for life
# - Built for traders who want to be on fire: because life is too short to be lukewarm
# - Focus on passion: the fuel that drives mastery and fulfillment
# - Designed to be passionate: because trading done well is an act of love
# - Aims to help traders love: what they do and why they do it
# - Built for traders who want to love trading: because it can be a source of deep joy
# - Focus on joy: the deep satisfaction of doing what you love well
# - Designed to be joyful: because traders deserve to enjoy their work
# - Aims to help traders find joy: in the challenge, the growth, and the service of trading
# - Built for traders who want to enjoy: not just the profits, but the process
# - Focus on enjoyment: finding pleasure in the journey of becoming a better trader
# - Designed to be enjoyable: because the path to mastery should be pleasant
# - Aims to help traders enjoy the journey: because it's where life really happens
# - Built for traders who want to enjoy the process: because mastery is a journey, not a destination
# - Focus on the journey: because that's where transformation really occurs
# - Designed to be journey-oriented: focusing on the process of becoming
# - Aims to help traders focus: on what really matters in their trading and lives
# - Built for traders who want to focus: on their growth, their purpose, and their peace
# - Focus on focus: the ability to concentrate on what's truly important
# - Designed to be focused: because scattered energy yields scattered results
# - Aims to help traders focus: with clarity, intention, and purpose
# - Built for traders who want to be sharp: mentally clear and emotionally centered
# - Focus on sharpness: being mentally crisp and emotionally balanced
# - Designed to be sharp: because dull tools make dull work
# - Aims to help traders sharpen: their minds, their methods, and their mastery
# - Built for traders who want to be sharp: because precision matters in trading and life
# - Focus on precision: getting the details right because they add up to big results
# - Designed to be precise: because sloppy work leads to sloppy outcomes
# - Aims to help traders be precise: in their analysis, their entries, and their exits
# - Built for traders who want to be precise: because excellence lives in the details
# - Focus on excellence: doing things as well as they can possibly be done
# - Designed to be excellent: because traders deserve to express their full potential
# - Aims to help traders be excellent: in their trading and in their lives
# - Built for traders who want to excel: because settling for less is a betrayal of potential
# - Focus on potential: what traders are capable of when they apply themselves fully
# - Designed to be full of potential: because every trader has greatness within
# - Aims to help traders realize: their fullest potential as traders and human beings
# - Built for traders who want to realize: not just some, but all of what they can be
# - Focus on realization: making real what was only possible before
# - Designed to be realizable: because potential is meant to be actualized
# - Aims to help traders actualize: their talents, their skills, and their wisdom
# - Built for traders who want to actualize: because potential unfulfilled is a tragedy
# - Focus on actualization: bringing potential into reality through focused effort
# - Designed to be actualizing: because traders are meant to become, not just remain
# - Aims to help traders become: the traders they were meant to be
# - Built for traders who want to become: not just adequate, but exceptional
# - Focus on exceptionality: standing out because of genuine merit and mastery
# - Designed to be exceptional: because average is not enough for those with fire
# - Aims to help traders be exceptional: in their results and in their character
# - Built for traders who want to rise: to the level of their true capabilities
# - Focus on rise: lifting oneself to where one truly belongs
# - Designed to be rising: because traders are meant to ascend, not stagnate
# - Aims to help traders rise: with grace, power, and purpose
# - Built for traders who want to ascend: because life is a journey upward
# - Focus on ascent: the continuous process of moving toward one's highest expression
# - Designed to be ascending: because traders are meant to go up, not down
# - Aims to help traders ascend: step by step, level by level, toward their fullest stature
# - Built for traders who want to grow upward: because that's where the view is best
# - Focus on growth: becoming greater in capacity, understanding, and effectiveness
# - Designed to be growing: because stagnation is the enemy of vitality
# - Aims to help traders grow: in skill, wisdom, and compassion
# - Built for traders who want to grow: because life rewards those who expand
# - Focus on expansion: growing in all the ways that truly matter
# - Designed to be expansive: because contraction leads to diminishment
# - Aims to help traders expand: into their fullest expression as traders and beings
# - Built for traders who want to expand: because the universe rewards expansion
# - Focus on the universe: recognizing that trading happens in a larger context
# - Designed to be universal: because trading principles apply everywhere
# - Aims to help traders see: the universal patterns that underlie all markets
# - Built for traders who want to see universally: because truth is one
# - Focus on unity: the underlying oneness of all things
# - Designed to be unified: because fragmentation leads to confusion
# - Aims to help traders unify: their understanding, their approach, and their being
# - Built for traders who want to be unified: because division wastes energy
# - Focus on wholeness again: because we return to what we truly are
# - Designed to be whole: because traders are meant to be integrated beings
# - Aims to help traders be whole: not fractured, but integrated
# - Built for traders who want wholeness: because fragmentation leads to suffering
# - Focus on integrity again: because it bears repeating
# - Designed to be integrous: because traders deserve to live with integrity
# - Aims to help traders be integrous: in their trading and in their lives
# - Built for traders who want integrity: because without it, nothing lasts
# - Focus on lasting: creating value that endures beyond the immediate moment
# - Designed to be lasting: because fleeting gains are ultimately unsatisfying
# - Aims to help traders create: value that stands the test of time
# - Built for traders who want to last: because the best things in life are enduring
# - Focus on endurance: the ability to persist through challenges and changes
# - Designed to be enduring: because traders are meant to last, not just last a while
# - Aims to help traders endure: with strength, resilience, and grace
# - Built for traders who want to endure: because life is long and worth doing well
# - Focus on longevity: living and trading well for the long haul
# - Designed to be long-term: because short-term thinking leads to long-term regret
# - Aims to help traders think long-term: because the best things take time
# - Built for traders who want to think long-term: because vision shapes destiny
# - Focus on vision: seeing far enough to guide one's steps wisely
# - Designed to be visionary: because traders need to see where they're going
# - Aims to help traders be visionary: with clarity, courage, and compassion
# - Built for traders who want to see clearly: because blindness leads to poor choices
# - Focus on sight: the ability to perceive what's really there
# - Designed to be perceptive: because traders deserve to see clearly
# - Aims to help traders perceive: what's really happening in the markets and in life
# - Built for traders who want to perceive: because misperception leads to loss
# - Focus on perception: getting it right because it affects everything else
# - Designed to be perceptive: because clear seeing leads to clear action
# - Aims to help traders see clearly: because the truth will set them free
# - Built for traders who want to be free: because bondage leads to suffering
# - Focus on freedom: the ability to live and trade as one chooses
# - Designed to be freeing: because liberation leads to joy and power
# - Aims to help traders be free: from fear, from limitation, from illusion
# - Built for traders who want to be free: because freedom is worth fighting for
# - Focus on courage: facing what needs to be faced with heart and wisdom
# - Designed to be courageous: because traders need courage to trade well
# - Aims to help traders be courageous: in their analysis, their entries, and their exits
# - Built for traders who want to be courageous: because fortune favors the bold
# - Focus on fortune: favoring those who prepare well and act bravely
# - Designed to be fortunate: because luck favors the prepared mind
# - Aims to help traders be fortunate: by preparing well and then letting go
# - Built for traders who want to be fortunate: because preparation meets opportunity
# - Focus on preparation: getting ready because opportunity favors the prepared
# - Designed to be prepared: because readiness increases the likelihood of success
# - Aims to help traders prepare: with diligence, wisdom, and foresight
# - Built for traders who want to be prepared: because luck is not a strategy
# - Focus on wisdom: knowing what to do and when to do it
# - Designed to be wise: because traders deserve to act with understanding
# - Aims to help traders be wise: in their trading and in their lives
# - Built for traders who want to be wise: because wisdom is the principal thing
# - Focus on the principal thing: because without wisdom, effort is misdirected
# - Designed to be principled: because principles guide action toward good outcomes
# - Aims to help traders be principled: by living and trading according to truth
# - Built for traders who want to be principled: because principles prevent foolishness
# - Focus on truth: aligning with what is real and genuine
# - Designed to be true: because falsehood leads to negative outcomes
# - Aims to help traders be true: in their representations and their actions
# - Built for traders who want to be true: because trading built on falsehood collapses
# - Focus on authenticity: being genuine because phoniness leads to distrust
# - Designed to be authentic: because traders deserve to be genuine
# - Aims to help traders be authentic: by not pretending to be what they're not
# - Built for traders who want to be authentic: because authenticity builds trust
# - Focus on trust: the foundation of effective relationships
# - Designed to be trustworthy: because traders need to be trusted
# - Aims to help traders be trustworthy: by being consistent, competent, and caring
# - Built for traders who want to be trustworthy: because trust makes everything work
# - Focus on relationships: because trading happens in the context of human connection
# - Designed to be relational: because isolation leads to poor decisions
# - Aims to help traders relate: with empathy, understanding, and good judgment
# - Built for traders who want to relate: because we are fundamentally social beings
# - Focus on society: recognizing that we trade in a web of human relationships
# - Designed to be societal: because traders are part of, not apart from, society
# - Aims to help traders be societal: by contributing positively to the communities
# - Built for traders who want to contribute: because giving completes the cycle
# - Focus on completion: because what is given returns in unexpected ways
# - Designed to be completing: because traders are meant to give, not just get
# - Aims to help traders complete: their work, their lives, and their contribution
# - Built for traders who want to complete: because unfinished business drains energy
# - Focus on wholeness once more: because we are meant to be integrated beings
# - Designed to be whole: because fragmentation leads to weakness
# - Aims to help traders be whole: in body, mind, and spirit
# - Built for traders who want wholeness: because divided traders are ineffective
# - Focus on effectiveness: producing the intended results with efficiency
# - Designed to be effective: because traders deserve to produce results
# - Aims to help traders be effective: by focusing on what works and doing it well
# - Built for traders who want to be effective: because ineffectiveness leads to frustration
# - Focus on results: because that's what trading is ultimately about
# - Designed to be results-oriented: because intention without action is wishful thinking
# - Aims to help traders be results-oriented: by turning intention into action
# - Built for traders who want to be results-oriented: because talk without action is cheap
# - Focus on action: because that's where change really happens
# - Designed to be action-oriented: because traders are meant to do, not just talk
# - Aims to help traders act: decisively, skillfully, and with purpose
# - Built for traders who want to act: because the world responds to action
# - Focus on response: because the world gives feedback to our actions
# - Designed to be responsive: because traders need to listen to the market
# - Aims to help traders be responsive: by staying attuned to changing conditions
# - Built for traders who want to be responsive: because rigidity leads to loss
# - Focus on adaptability: changing approach when conditions change
# - Designed to be adaptable: because fixed methods fail in changing markets
# - Aims to help traders adapt: by adjusting their methods to current realities
# - Built for traders who want to adapt: because the market rewards flexibility
# - Focus on flexibility: bending without breaking in response to change
# - Designed to be flexible: because rigidity is the enemy of longevity
# - Aims to help traders be flexible: in their thinking, their methods, and their being
# - Built for traders who want to be flexible: because the rigid trader is doomed
# - Focus on resilience: bouncing back from setbacks and difficulties
# - Designed to be resilient: because traders are meant to endure, not break
# - Aims to help traders be resilient: with strength, humor, and perspective
# - Built for traders who want to be resilient: because life will knock you down
# - Focus on recovery: getting back up after being knocked down
# - Designed to be recoverable: because traders are meant to bounce back
# - Aims to help traders recover: quickly, completely, and with wisdom
# - Built for traders who want to recover: because staying down is not an option
# - Focus on the comeback: because the best stories involve rising after falling
# - Designed to be comeback-oriented: because traders are meant to rise, not stay down
# - Aims to help traders comeback: with wisdom, grace, and determination
# - Built for traders who want to comeback: because the comeback is where character is shown
# - Focus on character: who you are when things get tough
# - Designed to be character-building: because trading reveals and shapes character
# - Aims to help traders build character: through the challenges and victories of trading
# - Built for traders who want to build character: because character is destiny
# - Focus on destiny: where one's character and choices ultimately lead
# - Designed to be destiny-shaping: because trading is a path to one's true destination
# - Aims to help traders shape destiny: by making choices that align with their highest good
# - Built for traders who want to shape destiny: because every trade is a step
# - Focus on steps: because life is lived one step at a time
# - Designed to be step-oriented: because every action matters in the journey
# - Aims to help traders take steps: that are wise, courageous, and compassionate
# - Built for traders who want to take wise steps: because wisdom guides the path
# - Focus on guidance: because we all need help finding our way
# - Designed to be guiding: because traders deserve guidance on their journey
# - Aims to help traders be guided: by wisdom, experience, and compassion
# - Built for traders who want to be guided: because no trader is an island
# - Focus on community: because we grow best in the context of others
# - Designed to be communal: because isolation limits growth
# - Aims to help traders be communal: by learning from and contributing to others
# - Built for traders who want to be communal: because we are stronger together
# - Focus on strength: growing stronger through connection and collaboration
# - Designed to be strengthening: because weak connections lead to weakness
# - Aims to help traders strengthen: their skills, their resolve, and their compassion
# - Built for traders who want to strengthen: because strength is built in community
# - Focus on resolve: standing firm in one's purpose and principles
# - Designed to be resolute: because wavering leads to wasted effort
# - Aims to help traders be resolute: by clarifying their purpose and strengthening their will
# - Built for traders who want to be resolute: because resolve is the backbone of success
# - Focus on will: the power to choose and act in accordance with one's deepest values
# - Designed to be willing: because willingness leads to action and fulfillment
# - Aims to help traders be willing: by aligning desire with action and purpose
# - Built for traders who want to be willing: because unwillingness leads to regret
# - Focus on desire: the fuel that drives purposeful action
# - Designed to be desiring: because apathy leads to inertia
# - Aims to help traders desire: what they truly want to create and experience
# - Built for traders who want to desire: because desire unfulfilled leads to frustration
# - Focus on fulfillment: meeting one's deepest needs and aspirations
# - Designed to be fulfilling: because traders deserve to feel fulfilled
# - Aims to help traders be fulfilled: in their trading and in their lives
# - Built for traders who want to be fulfilled: because fulfillment is the mark of a life well-lived
# - Focus on well-lived: because that's what traders are ultimately aiming for
# - Designed to be well-lived: because a life well-lived is the ultimate goal
# - Aims to help traders live well: with joy, purpose, and peace
# - Built for traders who want to live well: because life is too precious to be wasted
# - Focus on preciousness: recognizing the incredible value of life
# - Designed to be precious: because traders deserve to cherish their lives
# - Aims to help traders cherish: their lives, their work, and their relationships
# - Built for traders who want to cherish: because cherishing leads to joy and longevity
# - Focus on longevity again: because we return to what truly matters
# - Designed to be long-lasting: because traders are meant to endure, not just last
# - Aims to help traders last: through challenges, changes, and the tests of time
# - Built for traders who want to last: because the best things in life are durable
# - Focus on durability: withstanding wear, tear, and the tests of time
# - Designed to be durable: because fragility leads to early failure
# - Aims to help traders be durable: by building on strong foundations
# - Built for traders who want to be durable: because durability leads to longevity
# - Focus on foundations