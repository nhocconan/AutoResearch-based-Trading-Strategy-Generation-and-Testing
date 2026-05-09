#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_Pivot_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily Camarilla pivot levels (based on previous day)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    # Calculate pivots using previous day's data
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    # R3 and S3 levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and daily uptrend
            if (price > r3_aligned[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and  # Daily uptrend (EMA rising)
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and daily downtrend
            elif (price < s3_aligned[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and  # Daily downtrend (EMA falling)
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price closes below pivot or daily trend changes
            if (price < pivot[i] or 
                ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price closes above pivot or daily trend changes
            if (price > pivot[i] or 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals