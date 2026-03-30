#!/usr/bin/env python3
"""
Experiment #028: 1d KAMA Trend + RSI Momentum + Choppiness Regime + 1w Confirmation

HYPOTHESIS: KAMA adapts to volatility regime changes, making it effective in both
bull (breakouts) and bear (breakdowns). RSI(14) extremes (<30, >70) provide 
momentum-based entries. Choppiness Index filters out the 50% of time markets
are range-bound, avoiding 2022-style whipsaws. Weekly KAMA provides higher-
timeframe trend alignment for better entry quality.

WHY IT WORKS IN BULL AND BEAR: KAMA is adaptive - fast in trending markets,
slow in volatile ones. Short entries on RSI>70 in downtrending markets catches
bear rallies. Long entries on RSI<30 in uptrending markets catches reversals.

TARGET: 60-120 total trades over 4 years (15-30/year). HARD MAX: 150.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio (ER)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period)
    for i in range(n - period):
        for j in range(period):
            volatility[i] += abs(close[i + j + 1] - close[i + j])
    
    er = np.zeros(n)
    er[period:] = direction / np.maximum(volatility, 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_const = 2 / (fast + 1)
    slow_const = 2 / (slow + 1)
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    
    kama = np.full(n, np.nan)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    
    rs = np.divide(avg_gain, np.maximum(avg_loss, 1e-10))
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly KAMA for trend direction (slower = more reliable)
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Local 1d indicators
    kama_1d = calculate_kama(close, period=10)
    rsi_14 = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 50  # Need enough for KAMA(10) + Choppiness(14)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d[i]) or np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 1w KAMA TREND DIRECTION ===
        weekly_uptrend = kama_1w_aligned[i] > kama_1w_aligned[i - 1] if i > 0 else False
        weekly_downtrend = kama_1w_aligned[i] < kama_1w_aligned[i - 1] if i > 0 else False
        
        # === 1d KAMA DIRECTION ===
        daily_kama_up = kama_1d[i] > kama_1d[i - 1] if i > 1 else False
        daily_kama_down = kama_1d[i] < kama_1d[i - 1] if i > 1 else False
        
        # === REGIME: Skip if too choppy (CHOP > 61.8) ===
        is_choppy = chop[i] > 61.8
        
        # === RSI MOMENTUM ===
        rsi_oversold = rsi_14[i] < 32
        rsi_overbought = rsi_14[i] > 68
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # LONG: Weekly uptrend + Daily KAMA rising + RSI oversold
            if weekly_uptrend and daily_kama_up and rsi_oversold:
                desired_signal = SIZE
            
            # SHORT: Weekly downtrend + Daily KAMA falling + RSI overbought
            elif weekly_downtrend and daily_kama_down and rsi_overbought:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (3.0 ATR for daily = more room for noise) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLDING PERIOD EXIT (min 5 bars = 5 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 5:
            # Exit if KAMA flips against position
            if position_side > 0 and not daily_kama_up:
                desired_signal = 0.0
            if position_side < 0 and not daily_kama_down:
                desired_signal = 0.0
        
        # === CHOP EXIT (if market becomes choppy while in position) ===
        if in_position and is_choppy:
            # Reduce position in choppy markets
            if abs(desired_signal) == SIZE:
                desired_signal = SIZE / 2  # Half position in chop
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals