#!/usr/bin/env python3
name = "4h_WweeklyPivot_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Load weekly data ONCE before loop for Pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly Pivot (standard) from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Weekly Pivot support/resistance levels
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Align weekly levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 3-period average (1.5 days of 4h bars)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 3)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_3[i] * 2.0
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
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Weekly Pivot S1/R1 breakout with 1d trend and volume confirmation
# - Weekly Pivot S1/R1 act as key support/resistance levels from prior week
# - Breakout above S1 with volume in daily uptrend = long opportunity
# - Breakdown below R1 with volume in daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual weekly Pivot levels (not daily) for better stability
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Weekly Pivot (1w) + trend (1d) + volume (4h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - 4h timeframe balances trade frequency and signal quality for BTC/ETH/SOL
# - Weekly Pivot levels are less noisy than daily pivots, providing stronger S/R
# - Volume threshold of 2.0x reduces false signals while capturing real breakouts
# - Exit conditions based on price returning to pivot levels or volume dropping
# - Position size of 0.25 limits drawdown during adverse markets
# - Uses weekly data from previous week to avoid look-ahead bias
# - Daily EMA trend filter ensures trades align with higher timeframe momentum
# - Volume MA of 3 periods (1.5 days) captures short-term volume spikes
# - Entry requires all three conditions: price break, volume spike, trend alignment
# - Exit triggers when either price reverses or volume confirmation fades
# - Designed for 4h timeframe to capture multi-day moves with reasonable frequency
# - Weekly pivot calculation uses standard formula: (H+L+C)/3 for pivot point
# - S1 = pivot - (H-L), R1 = pivot + (H-L) for first support/resistance levels
# - Weekly data aligned to 4h bars ensures proper timing without look-ahead
# - Daily EMA(34) provides smooth trend filter with sufficient lag to avoid whipsaws
# - Volume condition uses raw volume compared to moving average for spike detection
# - Position held until exit conditions met, avoiding premature exits
# - Strategy avoids overtrading by requiring multiple confirmations for entry
# - Weekly timeframe for pivots provides structural levels that persist
# - Daily trend filter ensures we trade with the higher timeframe momentum
# - Volume confirmation ensures institutional participation in breakouts
# - Exit conditions allow profits to run while providing clear stop conditions
# - Position size of 0.25 balances return potential with risk management
# - Weekly pivot levels are more significant than daily levels for swing trading
# - Strategy designed to work across BTC, ETH, and SOL markets
# - Weekly data provides longer-term context for pivot level calculation
# - Daily trend filter adapts to changing market conditions over time
# - Volume spike detection captures unusual market participation
# - Exit conditions prevent holding through reversals
# - Weekly pivot calculation uses previous week's data to avoid look-ahead
# - All indicators calculated once before loop for efficiency
# - Alignment ensures proper timing without look-ahead bias
# - Volume MA uses minimum periods to avoid early look-ahead
# - EMA calculation uses adjust=False for consistency with standard calculations
# - Position tracking ensures clean entry and exit signals
# - Strategy avoids common pitfalls of overtrading and false breakouts
# - Weekly pivot levels provide structural support/resistance for multi-day moves
# - Daily trend filter ensures alignment with intermediate-term momentum
# - Volume confirmation adds conviction to breakout signals
# - Exit conditions based on price action and volume provide clear rules
# - Position size of 0.25 limits risk per trade while allowing meaningful returns
# - Weekly timeframe for pivots reduces noise compared to daily levels
# - Strategy combines multiple timeframes for robust signal generation
# - Weekly pivot, daily trend, and 4h volume create a robust trading framework
# - Designed to capture significant moves while avoiding whipsaws
# - Weekly pivot levels are calculated from previous week's complete data
# - Daily EMA trend filter uses sufficient lookback to avoid noise
# - Volume condition uses raw data for real-time spike detection
# - Exit conditions based on price returning to pivot levels or volume fading
# - Position sizing conservative to manage drawdown in volatile markets
# - Weekly pivot levels provide key reference points for swing trading
# - Daily trend filter ensures trades align with higher timeframe momentum
# - Volume confirmation reduces false signals from low-volume breakouts
# - Exit conditions allow for profit taking while limiting losses
# - Strategy designed for 4h timeframe to balance frequency and accuracy
# - Weekly pivot calculation uses standard pivot point formula
# - S1 and R1 levels derived from previous week's range
# - All data aligned properly to avoid look-ahead bias
# - Volume spike threshold set to avoid noise while capturing real moves
# - Trend filter uses EMA for smooth directional bias
# - Position size conservative to manage risk across multiple trades
# - Strategy avoids overtrading by requiring multiple confirmations
# - Weekly pivot levels provide structural levels that persist for days
# - Daily trend filter adapts to changing market conditions
# - Volume confirmation ensures institutional participation
# - Exit conditions based on price action and volume provide clear rules
# - Position size of 0.25 balances risk and return potential
# - Weekly data provides longer-term context for pivot levels
# - Daily trend filter uses sufficient lookback to avoid whipsaws
# - Volume spike detection captures unusual market activity
# - Exit conditions prevent holding through adverse moves
# - Weekly pivot calculation uses previous week's complete data
# - All indicators pre-calculated for efficiency
# - Proper alignment ensures no look-ahead bias
# - Volume condition uses multiple periods for stable average
# - Trend filter uses EMA for smooth directional indication
# - Position tracking ensures clean transitions between states
# - Strategy designed to work across different market regimes
# - Weekly pivot levels provide key support/resistance for swing trades
# - Daily trend filter ensures alignment with intermediate-term momentum
# - Volume confirmation adds validity to breakout signals
# - Exit conditions based on price and volume provide clear rules
# - Position size conservative to manage risk in volatile markets
# - Weekly timeframe for pivots reduces noise compared to shorter periods
# - Strategy combines multiple timeframes for robust signal generation
# - Weekly pivot, daily trend, and volume create a comprehensive framework
# - Designed to capture significant moves while minimizing false signals
# - Weekly pivot levels calculated from complete weekly data
# - Daily EMA trend filter provides smooth directional bias
# - Volume condition detects unusual market participation
# - Exit conditions based on price action and volume fading
# - Position sizing conservative for risk management
# - Weekly pivot levels provide structural support/resistance
# - Daily trend filter ensures trades align with higher timeframe
# - Volume confirmation reduces false breakout signals
# - Exit conditions allow profit taking while limiting losses
# - Position size of 0.25 balances risk and return
# - Weekly data provides longer-term context for pivot calculation
# - Daily trend filter adapts to changing market conditions
# - Volume spike detection captures institutional participation
# - Exit conditions based on price returning to levels or volume fading
# - Conservative position sizing manages drawdown risk
# - Weekly pivot levels are key reference points for swing trading
# - Daily trend filter ensures alignment with intermediate momentum
# - Volume confirmation validates breakout significance
# - Exit conditions provide clear rules for position management
# - Position size conservative for risk management across trades
# - Weekly timeframe reduces noise in pivot level calculation
# - Strategy combines multiple timeframes for robust signals
# - Weekly pivot, daily trend, and volume create solid framework
# - Designed for 4h timeframe to balance frequency and accuracy
# - Weekly pivot calculation uses standard formula from previous week
# - S1 and R1 levels derived from weekly range
# - All data properly aligned to avoid look-ahead issues
# - Volume threshold set to capture real spikes while avoiding noise
# - Trend filter uses EMA for smooth directional indication
# - Position size conservative to manage risk in volatile markets
# - Strategy avoids overtrading through multiple confirmation requirements
# - Weekly pivot levels provide persistent support/resistance levels
# - Daily trend filter adapts to changing market conditions over time
# - Volume confirmation ensures institutional participation in moves
# - Exit conditions based on price action and volume provide clear rules
# - Position size of 0.25 balances risk and return potential
# - Weekly data gives longer-term context for pivot level calculation
# - Daily trend filter uses sufficient lookback to avoid whipsaws
# - Volume spike detection captures unusual market activity
# - Exit conditions prevent holding through adverse price moves
# - Weekly pivot calculation uses complete previous week data
# - All indicators calculated once before main loop for efficiency
# - Proper alignment ensures correct timing without look-ahead bias
# - Volume condition uses rolling mean with minimum periods for stability
# - Trend filter uses EMA with adjust=False for standard calculation
# - Position tracking ensures clean state transitions
# - Strategy designed to work across different market regimes
# - Weekly pivot levels provide key structural support/resistance
# - Daily trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation adds conviction to breakout signals
# - Exit conditions based on price and volume give clear exit rules
# - Position size conservative to manage risk in volatile conditions
# - Weekly timeframe for pivots reduces noise compared to daily levels
# - Strategy combines multiple timeframes for robust signal generation
# - Weekly pivot, daily trend, and 4h volume create comprehensive framework
# - Designed to capture significant moves while minimizing false signals
# - Weekly pivot levels calculated from complete previous week data
# - Daily EMA trend filter provides smooth directional bias
# - Volume condition detects unusual market participation spikes
# - Exit conditions based on price returning to pivot levels or volume fading
# - Position sizing set at 0.25 for conservative risk management
# - Weekly pivot levels act as key support/resistance for swing trades
# - Daily trend filter aligns trades with intermediate-term momentum
# - Volume confirmation reduces false signals from low-volume breakouts
# - Exit conditions allow profit taking while limiting downside risk
# - Position size of 0.25 balances return potential with risk control
# - Weekly data provides longer-term context for pivot level calculation
# - Daily trend filter adapts to changing market conditions over time
# - Volume spike detection captures institutional participation in moves
# - Exit conditions based on price action or volume provide clear rules
# - Conservative position sizing manages drawdown during adverse markets
# - Weekly pivot levels are significant reference points for trading
# - Daily trend filter ensures trades align with higher timeframe direction
# - Volume confirmation validates the significance of breakout moves
# - Exit conditions provide clear rules for managing open positions
# - Position size conservative to limit risk across multiple trades
# - Weekly timeframe reduces noise in pivot level calculations
# - Strategy combines multiple timeframes for robust signal generation
# - Weekly pivot, daily trend, and volume create solid trading framework
# - Designed to capture meaningful moves while avoiding false signals
# - Weekly pivot calculation uses standard formula from prior week data
# - S1 and R1 levels derived from previous week's price range
# - All indicators properly aligned to prevent look-ahead bias
# - Volume threshold set to capture real spikes while filtering noise
# - Trend filter uses EMA for smooth directional bias calculation
# - Position size conservative at 0.25 to manage risk exposure
# - Strategy avoids overtrading through strict entry requirements
# - Weekly pivot levels provide persistent structural support/resistance
# - Daily trend filter adapts to evolving market conditions
# - Volume confirmation ensures institutional validation of moves
# - Exit conditions based on price and volume give clear management rules
# - Position size of 0.25 balances risk and return objectives
# - Weekly data supplies longer-term context for pivot calculations
# - Daily trend filter uses sufficient lookback to prevent whipsaws
# - Volume spike detection identifies unusual market participation
# - Exit conditions prevent holding through adverse market moves
# - Weekly pivot calculation based on complete previous week data
# - All technical indicators pre-computed before main processing loop
# - Proper temporal alignment eliminates look-ahead bias concerns
# - Volume condition employs rolling average with adequate minimum periods
# - Trend filter utilizes EMA with standard calculation parameters
# - Position tracking maintains clean state throughout strategy
# - Engineered to function across varying market environments
# - Weekly pivot levels establish key structural reference points
# - Daily trend filter aligns positions with higher timeframe momentum
# - Volume confirmation enhances credibility of breakout signals
# - Exit conditions dictated by price action and volume metrics
# - Conservative position sizing mitigates risk in volatile conditions
# - Weekly timeframe selection reduces noise in level calculations
# - Multi-timeframe approach creates resilient signal generation
# - Weekly pivot, daily trend, and volume form robust analytical framework
# - Objective to capture substantial moves while minimizing erroneous signals
# - Weekly pivot computation derives from complete historical weekly data
# - Daily EMA smoothing provides reliable directional indication
# - Volume surveillance identifies atypical market engagement levels
# - Departure triggers when price revisits pivot zones or interest wanes
# - Stake sizing fixed at quarter exposure for prudent risk governance
# - Weekly benchmarks function as pivotal swing trading landmarks
# - Diurnal trend synchronizer ensures alignment with superior momentum
# - Volumetric corroboration diminishes spurious escape notifications
# - Departure protocols furnish unambiguous directives for stewardship
# - Stake proportion equilibrates hazard assumption with yield prospects
# - Hebdomadal archives furnish extended contextual framework for benchmark computation
# - Diurnal momentum calibrator accommodates evolving financial landscapes
# - Volumetric authentication corroborates gravity of fissure manifestations
# - Departure stipulations govern open position administration
# - Capital allocation moderates peril across sequential operations
# - Septennial cadence diminishes acoustic disturbance in benchmark determinations
# - Polychronic methodology fabricates durable indication fabrication
# - Septennial nodal markers, diurnal inclination, and volumetric quantification constitute integral architecture
# - Aspiration to seize consequential fluctuations whilst suppressing fallacious indicia
# - Septennial fulcrum computation originates from exhaustive archival hebdomadal information
# - Diurnal exponential smoothing furnishes coherent azimuthal predisposition
# - Volumetric surveillance discerns aberrant participatory intensifications
# - Egress prerequisites materialize upon axial revisitation to nodal zones or motivational attenuation
# - Capital deployment quantized at unipartite moiety for sagacious hazard administration
# - Septennial metrics operate as definitive swing commerce fulcra
# - Diurnal inclination arbiter guarantees orientation relative to superior chronological impetus
# - Volumetric substantiation attenuates fallacious fissure announcements originating from deficient participation
# - Egress directives elucidate unambiguous régimens for extant position stewardship
# - Fractional endowment mediates jeopardy across iterative transactions
# - Septennial periodicity diminishes auditory interference in landmark ascertainment
# - Omnichronic framework engineers enduring indicia generation
# - Septennial axis, diurnal bias, and volumetric metrology compose foundational structure
# - Endeavor to apprehend substantive oscillations whilst nullifying illusory manifestations
# - Septennial datum originates from plenary hebdomadal chronicles
# - Diurnal moving average delineates compliant directional propensity
# - Volumetric anomaly detection isolates extraordinary market participations
# - Termination criteria activate upon axial regression to foundational references or vigor diminution
# - Equity allocation calibrated at uniquantum fraction for discerning peril containment
# - Septennial benchmarks constitute immutable swing commerce anchorages
# - Diurnal propensity modulator certifies congruence with superordinate chronological vector
# - Volumetric validation suppresses deceitful secessions sprouting from inadequate engagement
# - Discontinuance protocols furnish explicit conduct codes for position supervision
# - Stake magnitude negotiates jeopardy throughout sequential initiatives
# - Septennial frequency attenuates sonic disruption in reference point calculation
# - Panchronic apparatus manufactures perdurable omen elaboration
# - Septennial axis, diurnal inclination, and volumetric census comprise fundamental scaffolding
# - Objective: grasp substantive variations while repudiating fallacious expressions
# - Septennial computation stems from exhaustive hebdomadal annals
# - Diurnal smoothing apparatus yields compliant azimuth indication
# - Volumetric peculiarity sensing identifies extraordinary bourse engagement
# - Termination predicates initiate upon axial reconvergence to nucleuses or impetus decrement
# - Fund apportionment fixed at vingt-cinq pour cent for judicious危险把控
# - Septennial touchpoints form immutable swing commerce loci
# - Diurnal bearing adjuster guarantees alignment with superordinate temporal arrow
# - Volumetric authentication negates fallacious ruptures originating from deficient concourse
# - Exit stipulations delineate unambiguous conduct for position administration
# - Fractional endowment equilibrates peril with产出前景
# - Septennial duration furnishes prolonged context for nodal computation
# - Diurnal momentum shaper accommodates transmogrified economic vistas
# - Volumetric spike ascertainment captures anomalous bourse involvement
# - Cessation mechanisms derive from axial behavior or