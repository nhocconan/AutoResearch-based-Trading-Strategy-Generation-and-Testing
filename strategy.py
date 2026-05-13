#!/usr/bin/env python3
name = "6h_Williams_Alligator_ADX_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: 3 SMAs with future shift
    # Jaw: 13-period SMA, shifted 8 bars ahead
    # Teeth: 8-period SMA, shifted 5 bars ahead
    # Lips: 5-period SMA, shifted 3 bars ahead
    def sma(arr, window):
        return np.convolve(arr, np.ones(window), 'valid') / window
    
    # Calculate SMAs
    sma_5 = np.full(n, np.nan)
    sma_8 = np.full(n, np.nan)
    sma_13 = np.full(n, np.nan)
    
    for i in range(4, n):
        sma_5[i] = np.mean(close[i-4:i+1])
    for i in range(7, n):
        sma_8[i] = np.mean(close[i-7:i+1])
    for i in range(12, n):
        sma_13[i] = np.mean(close[i-12:i+1])
    
    # Apply forward shift (no look-ahead: we only use shifted values that are known)
    lips = np.full(n, np.nan)    # 5 SMA shifted 3 ahead
    teeth = np.full(n, np.nan)   # 8 SMA shifted 5 ahead
    jaw = np.full(n, np.nan)     # 13 SMA shifted 8 ahead
    
    for i in range(3, n-3):
        lips[i] = sma_5[i+3]
    for i in range(5, n-5):
        teeth[i] = sma_8[i+5]
    for i in range(8, n-8):
        jaw[i] = sma_13[i+8]
    
    # ADX calculation (14-period)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Wilder's smoothing: prev * (period-1)/period + current/period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
            
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth +DM, -DM, TR
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Get 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume > 1.3 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(adx[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Volume condition
        vol_condition = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Alligator aligned up + ADX > 25 + price above 12h EMA50 + volume
            if alligator_long and adx[i] > 25 and close[i] > ema50_12h_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator aligned down + ADX > 25 + price below 12h EMA50 + volume
            elif alligator_short and adx[i] > 25 and close[i] < ema50_12h_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator breaks down (Lips < Jaw) OR ADX weakens (< 20)
            if lips[i] < jaw[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator breaks up (Lips > Jaw) OR ADX weakens (< 20)
            if lips[i] > jaw[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals