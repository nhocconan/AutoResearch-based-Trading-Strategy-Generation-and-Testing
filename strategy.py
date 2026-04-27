# 4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance in 1d timeframe.
# Breakout above R3 with volume spike and 1d EMA34 uptrend = long.
# Breakdown below S3 with volume spike and 1d EMA34 downtrend = short.
# Uses volume confirmation and trend filter to avoid false breakouts.
# Designed for 4h timeframe with 1d/1h confirmation to reduce overtrading.
# Target: 20-40 trades/year per symbol, low frequency to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day (using typical price)
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Previous day's range
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = PP + (H-L)*1.1/2, S3 = PP - (H-L)*1.1/2
    # Where PP = (H + L + C) / 3 (same as typical price)
    pp = typical_price
    r3 = pp + range_1d * 1.1 / 2
    s3 = pp - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d_34 = np.full(len(df_1d), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1d)):
        if i < 33:
            ema_1d_34[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_1d_34[i-1]):
                ema_1d_34[i] = np.mean(close_1d[i-33:i+1])
            else:
                ema_1d_34[i] = close_1d[i] * alpha + ema_1d_34[i-1] * (1 - alpha)
    
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Volume spike detector: current volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    volume_spike = np.full(n, False)
    volume_spike[20:] = volume[20:] > (vol_ma[20:] * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_1d_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + 1d EMA uptrend
            if (price > r3_aligned[i] and 
                volume_spike[i] and 
                ema_1d_34_aligned[i] > ema_1d_34_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume spike + 1d EMA downtrend
            elif (price < s3_aligned[i] and 
                  volume_spike[i] and 
                  ema_1d_34_aligned[i] < ema_1d_34_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below R3 or EMA turns down
            if (price < r3_aligned[i] or 
                ema_1d_34_aligned[i] < ema_1d_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above S3 or EMA turns up
            if (price > s3_aligned[i] or 
                ema_1d_34_aligned[i] > ema_1d_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0