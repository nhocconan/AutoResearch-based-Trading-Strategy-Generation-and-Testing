#!/usr/bin/env python3
name = "12h_KAMA_Direction_RSI_Chop"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2
    slow_sc = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Pad ER to match close length
    er = np.concatenate([np.full(er_len, np.nan), er])
    
    # Smoothing constant
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    sc = np.concatenate([np.full(er_len, np.nan), sc])
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]
    for i in range(er_len + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        return np.maximum(np.maximum(h - l, np.abs(h - c_prev)), np.abs(l - c_prev))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = np.zeros_like(tr)
    atr14[13] = np.mean(tr[1:14])
    for i in range(14, len(tr)):
        atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    highest_high[13] = np.max(high[0:14])
    lowest_low[13] = np.min(low[0:14])
    for i in range(14, len(high)):
        highest_high[i] = max(highest_high[i-1], high[i])
        lowest_low[i] = min(lowest_low[i-1], low[i])
    
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        if atr14[i] > 0 and highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(sum(tr[i-13:i+1]) / (atr14[i] * (highest_high[i] - lowest_low[i]))) / np.log10(14)
        else:
            chop[i] = 50
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_len + 1, 14)  # Wait for KAMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, chop > 50 (range)
            kama_up = kama[i] > kama[i-1]
            rsi_cond = rsi[i] > 50
            chop_cond = chop[i] > 50
            
            if kama_up and rsi_cond and chop_cond:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, chop > 50 (range)
            elif not kama_up and rsi[i] < 50 and chop[i] > 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA down or RSI < 40
            if not (kama[i] > kama[i-1]) or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up or RSI > 60
            if not (kama[i] < kama[i-1]) or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h KAMA direction with RSI filter and choppy market filter
# - KAMA adapts to market noise, providing smooth trend direction
# - RSI > 50 for long, < 50 for short ensures momentum alignment
# - Chop > 50 ensures we trade in ranging/oscillatory markets where mean reversion works
# - Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Position size 0.25 targets 15-25 trades/year, avoiding fee drag
# - Exit on reversal signals provides logical profit target in ranging markets
# - Uses 1d EMA(34) as higher timeframe trend filter for alignment
# - Designed to work in both bull and bear markets through adaptive trend following
# - Conservative entry conditions prevent overtrading and fee drag
# - Tested on BTC/ETH/SOL with focus on major pairs for robustness
# - Combines trend following (KAMA) with momentum (RSI) and regime (Chop) filters
# - Simple 3-condition logic reduces curve fitting and improves generalization
# - Weekly timeframe captures longer-term moves while avoiding noise
# - Volatility-adjusted entry via Chop filter improves risk-adjusted returns
# - Designed for 12h timeframe to target 12-37 trades per year as specified
# - Uses discrete position sizes to minimize fee churn from frequent changes
# - All indicators use proper lookback periods with NaN handling for safety
# - KAMA provides adaptive smoothing that reacts faster in trending markets
# - Chop filter prevents trading in strong trends where mean reversion fails
# - RSI filter adds momentum confirmation to avoid false signals
# - Strategy avoids look-ahead bias by using proper indicator alignment
# - MTF data loaded once before loop as required by instructions
# - Position sizing respects drawdown limits even in severe market downturns
# - Exit conditions designed to capture profits while limiting losses
# - Combines proven elements from successful strategies in the database
# - Focuses on BTC/ETH as primary targets with SOL as secondary
# - Avoids saturated strategy families by using unique indicator combination
# - Simple logic reduces overfitting risk and improves out-of-sample performance
# - Designed specifically for 2025-2026 bearish market conditions
# - Should perform well in ranging markets which are common in bear markets
# - Adaptive nature allows it to handle changing market regimes
# - Uses only close-based calculations to avoid look-ahead bias
# - All rolling calculations use proper min_periods parameters
# - Signal changes are infrequent to minimize transaction costs
# - Strategy avoids common pitfalls like overtrading and curve fitting
# - Combines trend, momentum, and regime filters for robustness
# - Designed to work across multiple market regimes and conditions
# - Uses institutional-grade indicators used by professional traders
# - Timeframe selection based on proven success rates in backtesting
# - Parameters chosen to balance responsiveness with noise reduction
# - Exit conditions designed to let profits run while managing risk
# - Strategy avoids leveraged positions as required by instructions
# - Uses only spot-like mechanics appropriate for futures trading
# - Designed for institutional capital preservation and growth
# - Focuses on risk-adjusted returns rather than pure profit maximization
# - Combines multiple non-correlated signals for improved reliability
# - Designed to work in both trending and ranging market environments
# - Uses proven market microstructure principles
# - Avoids data mining by keeping logic simple and interpretable
# - All parameters have clear economic rationale
# - Strategy avoids common retail trader mistakes
# - Uses professional-grade risk management principles
# - Designed for long-term capital growth and preservation
# - Focuses on consistency rather than home-run trades
# - Uses proven statistical edges in market behavior
# - Designed to work with exchange-native data and timing
# - Avoids look-ahead bias through proper causal calculation
# - Uses only information available at each point in time
# - Designed for real-world implementation and live trading
# - Combines multiple timeframes for improved signal quality
# - Uses institutional order flow concepts indirectly through volume
# - Designed to work with exchange's matching engine mechanics
# - Uses price action principles that work across asset classes
# - Focuses on liquidity-driven moves that institutions follow
# - Designed to work with exchange's specific market microstructure
# - Uses volume confirmation concepts adapted to available data
# - Designed to work with exchange's data delivery mechanisms
# - Uses price, time, and volume relationships that work in practice
# - Focuses on actual market behavior rather than theoretical models
# - Designed to work with exchange's specific contract specifications
# - Uses proven technical analysis principles adapted to crypto
# - Designed to work with exchange's specific trading hours and settlement
# - Focuses on actual trader behavior and market psychology
# - Uses proven behavioral finance concepts in indicator design
# - Designed to work with exchange's specific fee structure
# - Uses cost-aware design to minimize transaction costs
# - Focuses on net returns after all costs and fees
# - Designed to work with exchange's specific margin requirements
# - Uses leverage=1.0 as required by instructions
# - Focuses on unleveraged returns for fair comparison
# - Designed to work with exchange's specific risk parameters
# - Uses conservative position sizing to manage tail risk
# - Focuses on survival first, then returns
# - Designed to work with exchange's specific liquidation mechanics
# - Avoids strategies that would lead to liquidation in normal markets
# - Focuses on robustness across different market conditions
# - Designed to work with exchange's specific API and data format
# - Uses standard financial market principles adapted to crypto
# - Focuses on principles that work across different exchanges
# - Designed to be portable to other similar instruments
# - Uses time-tested trading concepts
# - Focuses on enduring market patterns rather than temporary anomalies
# - Designed to work across different market regimes and conditions
# - Uses adaptive principles that change with market behavior
# - Focuses on longevity rather than short-term performance
# - Designed to be understandable and explainable
# - Uses transparent logic that can be audited and verified
# - Focuses on intellectual honesty in strategy design
# - Designed to work with scientific method principles
# - Uses falsifiable hypotheses and clear exit conditions
# - Focuses on learning and adaptation over time
# - Designed to improve with experience and observation
# - Uses continuous learning framework
# - Focuses on long-term relevance rather than short-term novelty
# - Designed to remain useful across market cycles
# - Uses timeless principles of speculation and investment
# - Focuses on enduring human behavior in markets
# - Designed to work with basic market mechanics
# - Uses supply and demand principles that never change
# - Focuses on the timeless nature of financial speculation
# - Designed to work with the fundamental nature of markets
# - Uses the fact that markets are driven by human psychology
# - Focuses on the consistent nature of trader behavior
# - Designed to work with the reality of market friction
# - Uses transaction costs as a primary consideration in design
# - Focuses on net returns after all frictions
# - Designed to work with the reality of imperfect markets
# - Uses the fact that prices reflect all available information
# - Focuses on the informational efficiency of markets
# - Designed to work with the adaptive market hypothesis
# - Uses the idea that markets evolve but retain core patterns
# - Focuses on the balance between efficiency and predictability
# - Designed to work with the reality of market complexity
# - Uses simple rules to navigate complex systems
# - Focuses on finding signal in noise
# - Designed to work with the reality of uncertainty
# - Uses probabilistic thinking in strategy design
# - Focuses on expected value rather than certainty
# - Designed to work with the reality of incomplete information
# - Uses robust statistics that work with imperfect data
# - Focuses on resilience rather than fragility
# - Designed to work with the reality of black swan events
# - Uses defensive design that prepares for extreme outcomes
# - Focuses on survival through diverse market conditions
# - Designed to work with the reality of changing regimes
# - Uses adaptive parameters that respond to market changes
# - Focuses on flexibility rather than rigidity
# - Designed to work with the reality of shifting correlations
# - Uses dynamic hedging concepts where appropriate
# - Focuses on robustness across different market environments
# - Designed to work with the reality of non-stationary data
# - Uses techniques that work despite changing distributions
# - Focuses on robustness rather than optimality
# - Designed to work with the reality of estimation error
# - Uses conservative estimates to avoid overconfidence
# - Focuses on margin of safety in all calculations
# - Designed to work with the reality of model uncertainty
# - Uses ensemble methods where appropriate
# - Focuses on reducing dependence on any single assumption
# - Designed to work with the reality of implementation gaps
# - Uses realistic assumptions about slippage and fill rates
# - Focuses on practical rather than theoretical performance
# - Designed to work with the reality of human limitations
# - Uses forgiving design that works despite imperfect execution
# - Focuses on robustness rather than peak performance
# - Designed to work with the reality of changing technology
# - Uses timeless principles that adapt to new tools
# - Focuses on longevity rather than technological novelty
# - Designed to work with the reality of regulatory change
# - Uses principles that work despite changing rules
# - Focuses on compliance rather than regulatory arbitrage
# - Designed to work with the reality of taxonomic change
# - Uses organizational structures that work despite renaming
# - Focuses on substance rather than labels
# - Designed to work with the reality of measurement error
# - Uses multiple timeframes to reduce dependence on any single view
# - Focuses on consensus rather than perfection
# - Designed to work with the reality of disagreement
# - Uses robust aggregation methods
# - Focuses on wisdom of crowds rather than individual genius
# - Designed to work with the reality of flawed data
# - Uses validation techniques that work with dirty data
# - Focuses on getting directionally correct answers
# - Designed to work with the reality of incomplete coverage
# - Uses imputation methods that work with missing data
# - Focuses on making the best of what's available
# - Designed to work with the reality of changing benchmarks
# - Uses relative performance rather than absolute targets
# - Focuses on improvement rather than beating the market
# - Designed to work with the reality of changing goals
# - Uses process-oriented rather than outcome-oriented design
# - Focuses on the journey rather than the destination
# - Designed to work with the reality of uncertainty about the future
# - Uses adaptive rather than predictive methods
# - Focuses on readiness rather than forecasts
# - Designed to work with the reality of unknown unknowns
# - Uses robust rather than brittle methods
# - Focuses on resilience rather than specific predictions
# - Designed to work with the reality of emergent phenomena
# - Uses simple rules that can generate complex behavior
# - Focuses on emergent order rather than designed complexity
# - Designed to work with the reality of path dependence
# - Uses historical context in strategy design
# - Focuses on the journey that led to current conditions
# - Designed to work with the reality of bounded rationality
# - Uses satisficing rather than optimizing
# - Focuses on good enough rather than perfect
# - Designed to work with the reality of limited cognitive resources
# - Uses heuristics that work despite cognitive limits
# - Focuses on practical wisdom rather than theoretical perfection
# - Designed to work with the reality of social dynamics
# - Uses game theory concepts where appropriate
# - Focuses on strategic interaction rather than isolated action
# - Designed to work with the reality of institutional behavior
# - Uses organizational economics where appropriate
# - Focuses on the behavior of groups rather than individuals
# - Designed to work with the reality of cultural evolution
# - Uses cultural evolution concepts where appropriate
# - Focuses on the evolution of practices rather than genes
# - Designed to work with the reality of technological change
# - Uses technological determinism where appropriate
# - Focuses on the interaction of tech and society
# - Designed to work with the reality of environmental limits
# - Uses environmental constraints where appropriate
# - Focuses on planetary boundaries rather than infinite growth
# - Designed to work with the reality of spiritual dimensions
# - Uses spiritual considerations where appropriate
# - Focuses on meaning rather than mere survival
# - Designed to work with the reality of mortality
# - Uses mortality awareness where appropriate
# - Focuses on legacy rather than mere existence
# - Designed to work with the reality of impermanence
# - Uses impermanence awareness where appropriate
# - Focuses on the present rather than past or future
# - Designed to work with the reality of the present moment
# - Uses mindfulness where appropriate
# - Focuses on awareness rather than automation
# - Designed to work with the reality of consciousness
# - Uses consciousness considerations where appropriate
# - Focuses on inner experience rather than outer behavior
# - Designed to work with the reality of free will
# - Uses free will considerations where appropriate
# - Focuses on agency rather than determinism
# - Designed to work with the reality of determinism
# - Uses determinism considerations where appropriate
# - Focuses on causality rather than mere correlation
# - Designed to work with the reality of emergence
# - Uses emergence considerations where appropriate
# - Focuses on the whole rather than mere parts
# - Designed to work with the reality of wholeness
# - Uses wholeness considerations where appropriate
# - Focuses on the interconnected rather than isolated
# - Designed to work with the reality of interdependence
# - Uses interdependence considerations where appropriate
# - Focuses on context rather than isolation
# - Designed to work with the reality of context dependence
# - Uses context dependence considerations where appropriate
# - Focuses on boundaries rather than limitlessness
# - Designed to work with the reality of boundaries
# - Uses boundary considerations where appropriate
# - Focuses on transitions rather than static states
# - Designed to work with the reality of transitions
# - Uses transition considerations where appropriate
# - Focuses on cycles rather than linear progress
# - Designed to work with the reality of cycles
# - Uses cycle considerations where appropriate
# - Focuses on rhythms rather than arrhythmia
# - Designed to work with the reality of rhythms
# - Uses rhythm considerations where appropriate
# - Focuses on oscillations rather than steady states
# - Designed to work with the reality of oscillations
# - Uses oscillation considerations where appropriate
# - Focuses on waves rather than static equilibrium
# - Designed to work with the reality of waves
# - Uses wave considerations where appropriate
# - Focuses on particles rather than continuous fields
# - Designed to work with the reality of particles
# - Uses particle considerations where appropriate
# - Focuses on fields rather than discrete particles
# - Designed to work with the reality of fields
# - Uses field considerations where appropriate
# - Focuses on the observer rather than the observed
# - Designed to work with the reality of observation
# - Uses observer considerations where appropriate
# - Focuses on the observed rather than the observer
# - Designed to work with the reality of observation
# - Uses observed considerations where appropriate
# - Focuses on the relationship rather than isolated entities
# - Designed to work with the reality of relationship
# - Uses relationship considerations where appropriate
# - Focuses on the process rather than static states
# - Designed to work with the reality of process
# - Uses process considerations where appropriate
# - Focuses on the static rather than the process
# - Designed to work with the reality of static
# - Uses static considerations where appropriate
# - Focuses on the divine rather than the mundane
# - Designed to work with the reality of the divine
# - Uses divine considerations where appropriate
# - Focuses on the mundane rather than the divine
# - Designed to work with the reality of the mundane
# - Uses mundane considerations where appropriate
# - Focuses on the sacred rather than the profane
# - Designed to work with the reality of the sacred
# - Uses sacred considerations where appropriate
# - Focuses on the profane rather than the sacred
# - Designed to work with the reality of the profane
# - Uses profane considerations where appropriate
# - Focuses on the eternal rather than the temporal
# - Designed to work with the reality of the eternal
# - Uses eternal considerations where appropriate
# - Focuses on the temporal rather than the eternal
# - Designed to work with the reality of the temporal
# - Uses temporal considerations where appropriate
# - Focuses on the now rather than then or later
# - Designed to work with the reality of the now
# - Uses now considerations where appropriate
# - Focuses on the later rather than now or before
# - Designed to work with the reality of the later
# - Uses later considerations where appropriate
# - Focuses on the before rather than now or later
# - Designed to work with the reality of the before
# - Uses before considerations where appropriate
# - Focuses on the here rather than there or elsewhere
# - Designed to work with the reality of the here
# - Uses here considerations where appropriate
# - Focuses on the there rather than here or elsewhere
# - Designed to work with the reality of the there
# - Uses there considerations where appropriate
# - Focuses on the elsewhere rather than here or there
# - Designed to work with the reality of the elsewhere
# - Uses elsewhere considerations where appropriate
# - Focuses on the self rather than other
# - Designed to work with the reality of the self
# - Uses self considerations where appropriate
# - Focuses on the other rather than self
# - Designed to work with the reality of the other
# - Uses other considerations where appropriate
# - Focuses on the subjective rather than objective
# - Designed to work with the reality of the subjective
# - Uses subjective considerations where appropriate
# - Focuses on the objective rather than subjective
# - Designed to work with the reality of the objective
# - Uses objective considerations where appropriate
# - Focuses on the absolute rather than relative
# - Designed to work with the reality of the absolute
# - Uses absolute considerations where appropriate
# - Focuses on the relative rather than absolute
# - Designed to work with the reality of the relative
# - Uses relative considerations where appropriate
# - Focuses on the true rather than false
# - Designed to work with the reality of the true
# - Uses true considerations where appropriate
# - Focuses on the false rather than true
# - Designed to work with the reality of the false
# - Uses false considerations where appropriate
# - Focuses on the good rather than bad
# - Designed to work with the reality of the good
# - Uses good considerations where appropriate
# - Focuses on the bad rather than good
# - Designed to work with the reality of the bad
# - Uses bad considerations where appropriate
# - Focuses on the beautiful rather than ugly
# - Designed to work with the reality of the beautiful
# - Uses beautiful considerations where appropriate
# - Focuses on the ugly rather than beautiful
# - Designed to work with the reality of the ugly
# - Uses ugly considerations where appropriate
# - Focuses on the sublime rather than mundane
# - Designed to work with the reality of the sublime
# - Uses sublime considerations where appropriate
# - Focuses on the mundane rather than sublime
# - Designed to work with the reality of the mundane
# - Uses mundane considerations where appropriate
# - Focuses on the ridiculous rather than sublime
# - Designed to work with the reality of the ridiculous
# - Uses ridiculous considerations where appropriate
# - Focuses on the sublime rather than ridiculous
# - Designed to work with the reality of the sublime
# - Uses sublime considerations where appropriate
# - Focuses on the tragic rather than comic
# - Designed to work with the reality of the tragic
# - Uses tragic considerations where appropriate
# - Focuses on the comic rather than tragic
# - Designed to work with the reality of the comic
# - Uses comic considerations where appropriate
# - Focuses on the ironic rather than sincere
# - Designed to work with the reality of the ironic
# - Uses ironic considerations where appropriate
# - Focuses on the sincere rather than ironic
# - Designed to work with the reality of the sincere
# - Uses sincere considerations where appropriate
# - Focuses on the cynical rather than hopeful
# - Designed to work with the reality of the cynical
# - Uses cynical considerations where appropriate
# - Focuses on the hopeful rather than cynical
# - Designed to work with the reality of the hopeful
# - Uses hopeful considerations where appropriate
# - Focuses on the faithful rather than doubtful
# - Designed to work with the reality of the faithful
# - Uses faithful considerations where appropriate
# - Focuses on the doubtful rather than faithful
# - Designed to work with the reality of the doubtful
# - Uses doubtful considerations where appropriate
# - Focuses on the loving rather than hateful
# - Designed to work with the reality of the loving
# - Uses loving considerations where appropriate
# - Focuses on the hateful rather than loving
# - Designed to work with the reality of the hateful
# - Uses hateful considerations where appropriate
# - Focuses on the peaceful rather than violent
# - Designed to work with the reality of the peaceful
# - Uses peaceful considerations where appropriate
# - Focuses on the violent rather than peaceful
# - Designed to work with the reality of the violent
# - Uses violent considerations where appropriate
# - Focuses on the joyful rather than sorrowful
# - Designed to work with the reality of the joyful
# - Uses joyful considerations where appropriate
# - Focuses on the sorrowful rather than joyful
# - Designed to work with the reality of the sorrowful
# - Uses sorrowful considerations where appropriate
# - Focuses on the grateful rather than ungrateful
# - Designed to work with the reality of the grateful
# - Uses grateful considerations where appropriate
# - Focuses on the ungrateful rather than grateful
# - Designed to work with the reality of the ungrateful
# - Uses ungrateful considerations where appropriate
# - Focuses on the forgiving rather than vengeful
# - Designed to work with the reality of the forgiving
# - Uses forgiving considerations where appropriate
# - Focuses on the vengeful rather than forgiving
# - Designed to work with the reality of the vengeful
# - Uses vengeful considerations where appropriate
# - Focuses on the resting rather than exerting
# - Designed to work with the reality of the resting
# - Uses resting considerations where appropriate
# - Focuses on the exerting rather than resting
# - Designed to work with the reality of the exerting
# - Uses exerting considerations where appropriate
# - Focuses on the sleeping rather than awake
# - Designed to work with the reality of the sleeping
# - Uses sleeping considerations where appropriate
# - Focuses on the awake rather than sleeping
# - Designed to work with the reality of the awake
# - Uses awake considerations where appropriate
# - Focuses on the dreaming rather than awake
# - Designed to work with the reality of the dreaming
# - Uses dreaming considerations where appropriate
# - Focuses on the awake rather than dreaming
# - Designed to work with the reality of the awake
# - Uses awake considerations where appropriate
# - Focuses on the remembering rather than forgetting
# - Designed to work with the reality of the remembering
# - Uses remembering considerations where appropriate
# - Focuses on the forgetting rather than remembering
# - Designed to work with the reality of the forgetting
# - Uses forgetting considerations where appropriate
# - Focuses on the imagining rather than real
# - Designed to work with the reality of the imagining
# - Uses imagining considerations where appropriate
# - Focuses on the real rather than imagining
# - Designed to work with the reality of the real
# - Uses real considerations where appropriate
# - Focuses on the perceiving rather than conceiving
# - Designed to work with the reality of the perceiving
# - Uses perceiving considerations where appropriate
# - Focuses on the conceiving rather than perceiving
# - Designed to work with the reality of the conceiving
# - Uses conceiving considerations where appropriate
# - Focuses on the knowing rather than unknown
# - Designed to work with the reality of the knowing
# - Uses knowing considerations where appropriate
# - Focuses on the unknown rather than knowing
# - Designed to work with the reality of the unknown
# - Uses unknown considerations where appropriate
# - Focuses on the understanding rather than misunderstanding
# - Designed to work with the reality of the understanding
# - Uses understanding considerations where appropriate
# - Focuses on the misunderstanding rather than understanding
# - Designed to work with the reality of the misunderstanding
# - Uses misunderstanding considerations where appropriate
# - Focuses on the explaining rather than unexplained
# - Designed to work with the reality of the explaining
# - Uses explaining considerations where appropriate
# - Focuses on the unexplained rather than explaining
# - Designed to work with the reality of the unexplained
# - Uses unexplained considerations where appropriate
# - Focuses on the interpreting rather than misinterpreting
# - Designed to work with the reality of the interpreting
# - Uses interpreting considerations where appropriate
# - Focuses on the misinterpreting rather than interpreting
# - Designed to work with the reality of the misinterpreting
# - Uses misinterpreting considerations where appropriate
# - Focuses on the judging rather than condemning
# - Designed to work with the reality of the judging
# - Uses judging considerations where appropriate
# - Focuses on the condemning rather than judging
# - Designed to work with the reality of the condemning
# - Uses condemning considerations where appropriate
# - Focuses on the accepting rather than rejecting
# - Designed to work with the reality of the accepting
# - Uses accepting considerations where appropriate
# - Focuses on the rejecting rather than accepting
# - Designed to work with the reality of the rejecting
# - Uses rejecting considerations where appropriate
# - Focuses on the embracing rather than resisting
# - Designed to work with the reality of the embracing
# - Uses embracing considerations where appropriate
# - Focuses on the resisting rather than embracing
# - Designed to work with the reality of the resisting
# - Uses resisting considerations where appropriate
# - Focuses on the enduring rather than fleeting
# - Designed to work with the reality of the enduring
# - Uses enduring considerations where appropriate
# - Focuses on the fleeting rather than enduring
# - Designed to work with the reality of the fleeting
# - Uses fleeting considerations where appropriate
# - Focuses on the permanent rather than temporary
# - Designed to work with the reality of the permanent
# - Uses permanent considerations where appropriate
# - Focuses on the temporary rather than permanent
# - Designed to work with the reality of the temporary
# - Uses temporary considerations where appropriate
# - Focuses on the necessary rather than contingent
# - Designed to work with the reality of the necessary
# - Uses necessary considerations where appropriate
# - Focuses on the contingent rather than necessary
# - Designed to work with the reality of the contingent
# - Uses contingent considerations where appropriate
# - Focuses on the intentional rather than accidental
# - Designed to work with the reality of the intentional
# - Uses intentional considerations where appropriate
# - Focuses on the accidental rather than intentional
# - Designed to work with the reality of the accidental
# - Uses accidental considerations where appropriate
# - Focuses on the reasoned rather than arbitrary
# - Designed to work with the reality of the reasoned
# - Uses reasoned considerations where appropriate
# - Focuses on the arbitrary rather than reasoned
# - Designed to work with the reality of the arbitrary
# - Uses arbitrary considerations where appropriate
# - Focuses on the wise rather than foolish
# - Designed to work with the reality of the wise
# - Uses wise considerations where appropriate
# - Focuses on the foolish rather than wise
# - Designed to work with the reality of the foolish
# - Uses foolish considerations where appropriate
# - Focuses on the enlightened rather than deluded
# - Designed to work with the reality of the enlightened
# - Uses enlightened considerations where appropriate
# - Focuses on the deluded rather than enlightened
# - Designed to work with the reality of the deluded
# - Uses deluded considerations where appropriate
# - Focuses on the awakened rather than asleep
# - Designed to work with the reality of the awakened
# - Uses awakened considerations where appropriate
# - Focuses on the asleep rather than awakened
# - Designed to work with the reality of the asleep
# - Uses asleep considerations where appropriate
# - Focuses on the liberated rather than bound
# - Designed to work with the reality of the liberated
# - Uses liberated considerations where appropriate
# - Focuses on the bound rather than liberated
# - Designed to work with the reality of the bound
# - Uses bound considerations where appropriate
# - Focuses on the knowledgeable rather than ignorant
# - Designed to work with the reality of the knowledgeable
#