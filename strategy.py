#!/usr/bin/env python3
# Hypothesis: 12h Camarilla Pivot R3/S3 Breakout with 1d EMA Trend and Volume Spike
# Uses 12h Camarilla pivot levels (R3/S3) for breakout signals, filtered by 1d EMA(34) trend and volume spike
# Volume spike defined as current volume > 1.8x 20-period average volume
# Only takes longs when price > 12h Camarilla R3 AND 1d EMA(34) rising AND volume spike
# Only takes shorts when price < 12h Camarilla S3 AND 1d EMA(34) falling AND volume spike
# Exits when price crosses back below 12h Camarilla pivot point (PP) for longs or above PP for shorts
# Target: 20-40 trades per year with position size 0.28

name = "12h_Camarilla_R3S3_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]
    ema_rising = ema_34_1d > ema_34_1d_prev
    ema_falling = ema_34_1d < ema_34_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Get 12h data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    close_12h = df_12h['close']
    
    # Calculate Camarilla pivot levels for each 12h bar
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    pp_12h = (high_12h + low_12h + close_12h) / 3
    r3_12h = pp_12h + (high_12h - low_12h) * 1.1 / 2
    s3_12h = pp_12h - (high_12h - low_12h) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h.values)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h.values)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h.values)
    
    # Volume spike: current volume > 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(pp_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > 12h Camarilla R3 + 1d EMA rising + volume spike
            if (close[i] > r3_12h_aligned[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.28
                position = 1
            # Enter short: price < 12h Camarilla S3 + 1d EMA falling + volume spike
            elif (close[i] < s3_12h_aligned[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 12h Camarilla pivot point
            if close[i] < pp_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Exit short: price crosses above 12h Camarilla pivot point
            if close[i] > pp_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals