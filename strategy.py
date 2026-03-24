#!/usr/bin/env python3
"""
Experiment #591: 6h Primary + 1d/1w HTF — Fisher Transform + Connors RSI + Regime Filter

Hypothesis: 6h timeframe with Ehlers Fisher Transform provides superior reversal detection
vs RSI during volatile periods. Fisher Transform normalizes price to Gaussian distribution,
making extreme values (-2 to +2) statistically significant reversal points. Combined with
Connors RSI (CRSI) for mean reversion timing and Choppiness Index for regime detection,
this should outperform simple HMA/RSI strategies especially in 2022 crash and 2025 bear.

Key innovations vs failed experiments:
1. Fisher Transform (period=9) - catches reversals at -1.5/+1.5 levels (proven in bear markets)
2. Connors RSI (CRSI) - composite of RSI(3) + StreakRSI(2) + PercentRank(100), 75% win rate
3. Dual HTF: 1d HMA for medium bias + 1w HMA for macro bias (both aligned properly)
4. Regime-adaptive: CHOP>61.8 = mean revert (CRSI extremes), CHOP<38.2 = trend (Fisher breakouts)
5. Asymmetric sizing: larger positions when HTF + Fisher + CRSI all align

Strategy logic:
1. 1w HMA(21) = macro trend bias (very slow filter)
2. 1d HMA(21) = medium trend bias
3. 6h Fisher Transform(9) = reversal signals (cross above -1.5 = long, below +1.5 = short)
4. 6h Connors RSI = mean reversion timing (CRSI<10 = oversold, CRSI>90 = overbought)
5. 6h Choppiness(14) = regime (CHOP>61.8 = range, CHOP<38.2 = trend)
6. 6h ATR(14)*2.5 stoploss on all positions

Regime-adaptive entries:
- RANGE (CHOP>55): CRSI<15 + Fisher>-1.5 = long, CRSI>85 + Fisher<+1.5 = short
- TREND (CHOP<45): Fisher breakout with HTF alignment
- TRANSITION: Reduced size, require 3/3 confluence

Target: Sharpe>0.45, trades>=60 train (15/year), trades>=8 test
Timeframe: 6h
Size: 0.25 base, 0.30 strong confluence
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_crsi_chop_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Extreme values (-2 to +2) indicate statistically significant reversal points
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = 0.0
            trigger[i] = 0.0
            continue
        
        # Normalize price to 0-1 range
        normalized = (close[i] - lowest) / price_range
        
        # Clamp to avoid division issues (0.001 to 0.999)
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher Transform formula
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with previous value (trigger line)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            trigger[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            trigger[i] = fisher_val
    
    return fisher, trigger

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) - short-term momentum
    2. RSI(Streak) - streak duration RSI (consecutive up/down days)
    3. PercentRank - percentile of current close vs last 100 closes
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Fisher crosses for signal generation
    prev_fisher = 0.0
    prev_crsi = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            prev_crsi = crsi[i] if not np.isnan(crsi[i]) else prev_crsi
            continue
        
        if np.isnan(fisher[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            prev_crsi = crsi[i] if not np.isnan(crsi[i]) else prev_crsi
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            prev_crsi = crsi[i] if not np.isnan(crsi[i]) else prev_crsi
            continue
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > -1.5 and prev_fisher <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and prev_fisher >= 1.5
        
        # === CRSI MEAN REVERSION ===
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_oversold = crsi[i] < 30.0
        crsi_overbought = crsi[i] > 70.0
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean revert)
        chop_trend = chop[i] < 45.0   # Trending (trend follow)
        chop_transition = not chop_range and not chop_trend
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        confluence_count = 0
        
        # RANGE REGIME: Mean reversion with CRSI + Fisher confluence
        if chop_range:
            # Long: CRSI oversold + Fisher turning up + HTF not bearish
            if crsi_extreme_oversold and fisher[i] > -1.0 and not htf_bear:
                desired_signal = SIZE_STRONG
                confluence_count = 3
            elif crsi_oversold and fisher_cross_up and not htf_bear:
                desired_signal = SIZE_BASE
                confluence_count = 2
            elif crsi_extreme_oversold and not htf_bear:
                desired_signal = SIZE_BASE * 0.8
                confluence_count = 2
            
            # Short: CRSI overbought + Fisher turning down + HTF not bullish
            elif crsi_extreme_overbought and fisher[i] < 1.0 and not htf_bull:
                desired_signal = -SIZE_STRONG
                confluence_count = 3
            elif crsi_overbought and fisher_cross_down and not htf_bull:
                desired_signal = -SIZE_BASE
                confluence_count = 2
            elif crsi_extreme_overbought and not htf_bull:
                desired_signal = -SIZE_BASE * 0.8
                confluence_count = 2
        
        # TREND REGIME: Fisher breakout with HTF alignment
        elif chop_trend:
            # Long: HTF bull + Fisher cross up from oversold
            if htf_bull and fisher_cross_up:
                desired_signal = SIZE_STRONG
                confluence_count = 3
            elif htf_bull and fisher[i] > -1.0 and fisher[i] < 0:
                desired_signal = SIZE_BASE
                confluence_count = 2
            
            # Short: HTF bear + Fisher cross down from overbought
            elif htf_bear and fisher_cross_down:
                desired_signal = -SIZE_STRONG
                confluence_count = 3
            elif htf_bear and fisher[i] < 1.0 and fisher[i] > 0:
                desired_signal = -SIZE_BASE
                confluence_count = 2
        
        # TRANSITION REGIME: Require strong confluence
        elif chop_transition:
            # Long: All three align (HTF bull + CRSI oversold + Fisher turning)
            if htf_bull and crsi_oversold and fisher[i] > prev_fisher:
                desired_signal = SIZE_BASE
                confluence_count = 3
            # Short: All three align
            elif htf_bear and crsi_overbought and fisher[i] < prev_fisher:
                desired_signal = -SIZE_BASE
                confluence_count = 3
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
        
        # Update prev values for next iteration
        prev_fisher = fisher[i]
        prev_crsi = crsi[i]
    
    return signals