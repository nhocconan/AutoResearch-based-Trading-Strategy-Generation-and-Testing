#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability reversal/breakout zones. 1w EMA50 ensures alignment with higher timeframe trend.
# Volume spike (>2x 20 EMA) confirms institutional participation. Discrete sizing 0.25 limits risk.
# Works in bull/bear: trend filter prevents counter-trend entries. Target: 50-150 trades over 4 years.

name = "12h_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe (completed 1w bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C, H, L are from previous completed 1d bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's OHLC (avoid look-ahead)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # First value has no previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_r3_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2.0
    camarilla_s3_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot point (midpoint between R3 and S3) OR trend changes OR volume drops
            pivot_point = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if (close[i] < pivot_point or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot point OR trend changes OR volume drops
            pivot_point = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if (close[i] > pivot_point or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals