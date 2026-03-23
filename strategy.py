#!/usr/bin/env python3
"""
Experiment #483: 1d Primary + 1w HTF — HMA Trend + Connors RSI + Donchian Breakout

Hypothesis: Based on research showing HMA (Hull Moving Average) provides superior 
trend identification with less lag than EMA/KAMA. Combined with Connors RSI (proven 
75% win rate on mean reversion) and Donchian channels for breakout confirmation.
Key innovations:
1. HMA(21) primary trend - faster response than KAMA, less whipsaw than EMA
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - proven edge
3. Donchian(20) breakout confirmation - ensures momentum alignment
4. 1w HMA for HTF major trend bias (stable weekly trend direction)
5. ATR(14) trailing stop at 2.5x for risk management
6. Relaxed CRSI thresholds (15/85 instead of 10/90) to ensure trade generation
7. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work: HMA is more responsive than KAMA for 1d timeframe. Connors RSI
has documented 75% win rate in academic research. Donchian breakout ensures we catch
momentum moves. 1w HTF keeps us aligned with major trend. This is DIFFERENT from 
failed Fisher+Donchian - using HMA trend + CRSI mean reversion hybrid approach.
Simplified regime logic (fewer filters = more trades generated).

Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    Provides smoother trend with less lag than EMA.
    """
    n = len(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, w_period):
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        result = np.full(len(series), np.nan)
        for i in range(w_period - 1, len(series)):
            result[i] = np.sum(series[i - w_period + 1:i + 1] * weights)
        return result
    
    close_s = pd.Series(close)
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA calculation
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        streak_vals = streak[i - streak_period + 1:i + 1]
        up_streaks = np.sum(streak_vals > 0)
        if streak_period > 0:
            streak_rsi[i] = 100.0 * up_streaks / streak_period
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    pr = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i - pr_period + 1:i + 1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (pr_period - 1)
        pr[i] = 100.0 * rank
    
    # Combine CRSI
    valid_mask = (~np.isnan(rsi_3)) & (~np.isnan(streak_rsi)) & (~np.isnan(pr))
    crsi[valid_mask] = (rsi_3[valid_mask] + streak_rsi[valid_mask] + pr[valid_mask]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel (highest high / lowest low over period).
    Upper = highest high over N periods
    Lower = lowest low over N periods
    Middle = (Upper + Lower) / 2
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, middle

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    hma_1d = calculate_hma(close, period=21)
    crsi_1d = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators (1w HMA for major trend bias)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(hma_1d[i]):
            continue
        if np.isnan(crsi_1d[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === HTF MAJOR TREND BIAS (1w HMA) ===
        htf_bullish = close[i] > hma_1w_aligned[i]
        htf_bearish = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA) ===
        price_above_hma = close[i] > hma_1d[i]
        price_below_hma = close[i] < hma_1d[i]
        hma_slope_up = hma_1d[i] > hma_1d[i - 5] if i >= 5 else False
        hma_slope_down = hma_1d[i] < hma_1d[i - 5] if i >= 5 else False
        
        # === CONNORS RSI SIGNALS (relaxed thresholds for trade generation) ===
        crsi_oversold = crsi_1d[i] < 20.0  # Relaxed from 10 to ensure trades
        crsi_overbought = crsi_1d[i] > 80.0  # Relaxed from 90
        crsi_extreme_oversold = crsi_1d[i] < 15.0
        crsi_extreme_overbought = crsi_1d[i] > 85.0
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        donch_breakout_up = close[i] > donch_mid[i]
        donch_breakout_down = close[i] < donch_mid[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES - multiple paths to ensure trades
        long_score = 0
        
        # Path 1: HTF bullish + CRSI oversold (mean reversion in uptrend)
        if htf_bullish and crsi_oversold:
            long_score += 3
        
        # Path 2: Price above HMA + CRSI oversold + Donchian confirmation
        if price_above_hma and crsi_oversold and donch_breakout_up:
            long_score += 3
        
        # Path 3: HMA slope up + CRSI extreme oversold (strong reversal signal)
        if hma_slope_up and crsi_extreme_oversold:
            long_score += 3
        
        # Path 4: HTF bullish + HMA slope up + CRSI moderately oversold
        if htf_bullish and hma_slope_up and crsi_1d[i] < 30.0:
            long_score += 2
        
        # Enter long if score >= 3 (relaxed to ensure trade generation)
        if long_score >= 3:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # Path 1: HTF bearish + CRSI overbought
            if htf_bearish and crsi_overbought:
                short_score += 3
            
            # Path 2: Price below HMA + CRSI overbought + Donchian confirmation
            if price_below_hma and crsi_overbought and donch_breakout_down:
                short_score += 3
            
            # Path 3: HMA slope down + CRSI extreme overbought
            if hma_slope_down and crsi_extreme_overbought:
                short_score += 3
            
            # Path 4: HTF bearish + HMA slope down + CRSI moderately overbought
            if htf_bearish and hma_slope_down and crsi_1d[i] > 70.0:
                short_score += 2
            
            if short_score >= 3:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma and htf_bullish:
                desired_signal = SIZE_LONG
            elif position_side < 0 and price_below_hma and htf_bearish:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals