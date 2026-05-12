#!/usr/bin/env python3
"""
4H_CAMARILLA_R3_S3_BREAKOUT_1D_VOLUME_SPIKE_V2
Hypothesis: Refined version with tighter entry conditions to reduce trade frequency and improve edge.
- Uses stricter volume threshold (2.0x instead of 1.5x) to confirm institutional participation
- Adds ADX(14) > 25 filter to ensure trending market context
- Maintains core logic: Breakout of R3/S3 with volume confirmation, mean-reversion exits at opposite level
- Designed for ~15-25 trades/year on 4h to minimize fee drag while capturing strong moves
"""
name = "4H_CAMARILLA_R3_S3_BREAKOUT_1D_VOLUME_SPIKE_V2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Increased warmup for ADX calculation
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ADX(14) for trend strength filter
    # +DM, -DM, TR calculation
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) if not np.any(np.isnan(data[1:period])) else np.nan
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Calculate smoothed +DM, -DM, TR
    atr = WilderSmooth(tr, 14)
    plus_di = 100 * WilderSmooth(plus_dm, 14) / atr
    minus_di = 100 * WilderSmooth(minus_dm, 14) / atr
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = WilderSmooth(dx, 14)
    
    # Calculate Camarilla levels from previous day
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        camarilla_r3[i] = close[i-1] + (high[i-1] - low[i-1]) * 1.1 / 2
        camarilla_s3[i] = close[i-1] - (high[i-1] - low[i-1]) * 1.1 / 2
    
    # 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d / vol_ma_20d  # Current volume / 20-day average
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous day data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume spike (2.0x average)
            if (high[i] > camarilla_r3[i] and 
                vol_spike_aligned[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike (2.0x average)
            elif (low[i] < camarilla_s3[i] and 
                  vol_spike_aligned[i] > 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 (reversion to mean)
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (reversion to mean)
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals