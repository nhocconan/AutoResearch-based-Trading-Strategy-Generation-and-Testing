#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above R3 with 1d uptrend and volume > 2x average.
# Short when price breaks below S3 with 1d downtrend and volume > 2x average.
# Uses 4h timeframe for entries to limit trades (target: 20-50/year).
# Works in bull (breakouts in uptrend) and bear (breakouts in downtrend).

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align pivot levels to 4h (use previous day's levels)
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, 1d uptrend, volume spike
            if (close[i] > r3_4h[i] and 
                ema_34_1d_aligned[i] > np.roll(ema_34_1d_aligned, 1)[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, 1d downtrend, volume spike
            elif (close[i] < s3_4h[i] and 
                  ema_34_1d_aligned[i] < np.roll(ema_34_1d_aligned, 1)[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend reversal
            if (close[i] < s3_4h[i] or 
                ema_34_1d_aligned[i] < np.roll(ema_34_1d_aligned, 1)[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or trend reversal
            if (close[i] > r3_4h[i] or 
                ema_34_1d_aligned[i] > np.roll(ema_34_1d_aligned, 1)[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals