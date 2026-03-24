#!/usr/bin/env python3
"""
Experiment #075: 6h Primary + 12h/1d HTF — Connors RSI + Choppiness Regime + Dual HMA

Hypothesis: After 7 failed 6h experiments, the pattern shows 6h is too slow for pure mean-reversion
but too fast for pure trend-following. SOLUTION: Connors RSI (CRSI) for precise reversal entries,
filtered by Choppiness regime and dual-HTF HMA alignment.

Key innovations vs failed #071 (Donchian+HMA+RSI+Chop):
- Connors RSI (not standard RSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
  - More sensitive to short-term extremes, catches reversals earlier
  - Entry thresholds: CRSI < 15 (long), CRSI > 85 (short) — proven 75% win rate
- Choppiness regime switch: CHOP > 55 = range (allow mean-revert), CHOP < 45 = trend (skip counter-trend)
- Dual HTF filter: BOTH 12h HMA AND 1d HMA must align with trade direction
  - Prevents fighting major trend (the #1 killer in 2022 crash)
- Conservative sizing: 0.25 (25% capital) with 2.5x ATR trailing stop
- Target: 30-50 trades/year on 6h (lower than 4h to reduce fee drag)

Why this might work when #071 failed:
- #071 used Donchian breakouts (trend) + RSI (mean-revert) = conflicting signals
- This uses CRSI (pure mean-revert) + regime filter (only trade mean-revert in range)
- Dual HMA alignment prevents entering against major trend (2022 crash protection)

Target: Sharpe > 0.167 (beat current best), DD > -40%, trades >= 30 train, >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_chop_dual_hma_12h1d_v1"
timeframe = "6h"
leverage = 1.0

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    streak[:] = np.nan
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = 1.0
            j = i - 1
            while j > 0 and close[j] > close[j-1]:
                streak[i] += 1.0
                j -= 1
        elif close[i] < close[i-1]:
            streak[i] = -1.0
            j = i - 1
            while j > 0 and close[j] < close[j-1]:
                streak[i] -= 1.0
                j -= 1
        else:
            streak[i] = 0.0
    
    # Convert streak to RSI-like scale (0-100)
    # Positive streak = overbought, negative = oversold
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        if streak[i] >= period:
            streak_rsi[i] = 100.0
        elif streak[i] <= -period:
            streak_rsi[i] = 0.0
        else:
            # Map streak range [-period, +period] to [0, 100]
            streak_rsi[i] = 50.0 + (streak[i] / period) * 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI
    Measures current price change vs past period changes
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    for i in range(period, n):
        current_change = close[i] - close[i-1]
        past_changes = close[i-period+1:i] - close[i-period:i-1]
        
        count_lower = np.sum(past_changes < current_change)
        pct_rank[i] = (count_lower / period) * 100.0
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme readings (<10 or >90) indicate high-probability reversals
    """
    rsi = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi + streak_rsi + pct_rank) / 3.0
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    We use 55 as threshold for regime detection
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=34)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 6h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (Dual HMA alignment) ===
        # Both 12h and 1d must align for trade entry
        htf_bull = (close[i] > hma_12h_aligned[i]) and (close[i] > hma_1d_aligned[i])
        htf_bear = (close[i] < hma_12h_aligned[i]) and (close[i] < hma_1d_aligned[i])
        htf_neutral = not htf_bull and not htf_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range/choppy (allow mean-revert entries)
        # CHOP < 45 = trending (skip counter-trend, only follow trend)
        # 45-55 = transition (reduced position)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        is_transition = not is_choppy and not is_trending
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = oversold (long opportunity)
        # CRSI > 85 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === 6h HMA TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === DESIRED SIGNAL (CRSI + Regime + HTF) ===
        desired_signal = 0.0
        signal_strength = 1.0
        
        if is_choppy:
            # RANGE REGIME: Mean-revert on CRSI extremes
            # LONG: CRSI oversold + HTF not bearish
            if crsi_oversold and not htf_bear:
                desired_signal = SIZE
                if crsi_extreme_oversold:
                    signal_strength = 1.0
            # SHORT: CRSI overbought + HTF not bullish
            elif crsi_overbought and not htf_bull:
                desired_signal = -SIZE
                if crsi_extreme_overbought:
                    signal_strength = 1.0
        
        elif is_trending:
            # TREND REGIME: Only follow major trend on CRSI pullback
            # LONG: HTF bull + CRSI pullback (not extreme oversold) + 6h HMA bull
            if htf_bull and crsi[i] < 50.0 and hma_6h_bull:
                desired_signal = SIZE * 0.7
            # SHORT: HTF bear + CRSI pullback (not extreme overbought) + 6h HMA bear
            elif htf_bear and crsi[i] > 50.0 and hma_6h_bear:
                desired_signal = -SIZE * 0.7
        
        else:
            # TRANSITION REGIME: Reduced size, require extreme CRSI
            if crsi_extreme_oversold and not htf_bear:
                desired_signal = SIZE * 0.5
            elif crsi_extreme_overbought and not htf_bull:
                desired_signal = -SIZE * 0.5
        
        # Apply signal strength
        desired_signal = desired_signal * signal_strength
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals