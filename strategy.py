#!/usr/bin/env python3
"""
Experiment #021: 12h Williams %R + Choppiness + Volume Spike

HYPOTHESIS: Williams %R at extreme levels (<-80 or >-20) captures capitulation/
reversal points. Combined with Choppiness regime (>58 = range-bound = mean reversion
works) and volume spike confirmation, this identifies high-probability reversals.

WHY 12h: 3x slower than 4h = fewer trades = less fee drag.
Williams %R(14) on 12h = 7-day lookback = captures multi-day swings.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Buy oversold with price above EMA = fade panic dip
- Bear: Sell overbought with price below EMA = catch bear rallies

TARGET: 60-120 total trades over 4 years (15-30/year). HARD MAX: 150.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_chop_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator for overbought/oversold"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    willr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 0:
            willr[i] = -100.0 * (highest_high - close[i]) / range_hl
    
    return willr

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies ranging vs trending markets"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        sum_tr = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[0] - low[0]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            sum_tr += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * (np.log(sum_tr) / np.log(range_hl * period)) if range_hl > 0 else np.nan
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    willr_14 = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    stop_price = 0.0
    
    warmup = 100  # Buffer for alignments
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(willr_14[i]) or np.isnan(atr_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK ===
        is_choppy = chop_14[i] > 58.0  # Range-bound = mean reversion works
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === WILLIAMS %R EXTREMES ===
        willr_oversold = willr_14[i] < -80  # Extreme oversold
        willr_overbought = willr_14[i] > -20  # Extreme overbought
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Oversold + price above EMA (bull trend) + volume + choppy regime
            if willr_oversold and price_above_1d_ema and vol_spike and is_choppy:
                desired_signal = SIZE
            
            # SHORT: Overbought + price below EMA (bear trend) + volume + choppy regime
            elif willr_overbought and not price_above_1d_ema and vol_spike and is_choppy:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (2 bars = 1 day) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 2:
            # Exit on Williams %R reversal (returned to neutral)
            if position_side > 0 and willr_14[i] > -50:  # No longer oversold
                desired_signal = 0.0
            if position_side < 0 and willr_14[i] < -50:  # No longer overbought
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals