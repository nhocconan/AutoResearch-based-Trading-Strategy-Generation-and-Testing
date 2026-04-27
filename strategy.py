#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 Breakout with 1d EMA34 trend filter and volume spike
# Uses Camarilla pivot levels from 1d (R3/S3) for breakout entries
# Trend filter: price above/below 1d EMA34 to avoid counter-trend trades
# Volume confirmation: volume > 2x 20-period average to ensure conviction
# Target: 20-35 trades/year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1/2
    # S3 = Pivot - (H - L) * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 34-period EMA on daily close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Close breaks above R3 AND price above daily EMA34 (uptrend) AND volume spike
        if (close[i] > r3_aligned[i] and 
            close[i] > ema34_1d_aligned[i] and volume_filter[i]):
            signals[i] = 0.30
            position = 1
        # Short conditions: Close breaks below S3 AND price below daily EMA34 (downtrend) AND volume spike
        elif (close[i] < s3_aligned[i] and 
              close[i] < ema34_1d_aligned[i] and volume_filter[i]):
            signals[i] = -0.30
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0