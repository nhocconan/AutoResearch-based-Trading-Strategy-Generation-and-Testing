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
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily pivot levels from previous day (classic pivot point)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_ = high_prev - low_prev
    
    # Resistance and Support levels (R1/S1, R2/S2, R3/S3)
    r1 = pivot + (range_ * 1)
    s1 = pivot - (range_ * 1)
    r2 = pivot + (range_ * 2)
    s2 = pivot - (range_ * 2)
    r3 = pivot + (range_ * 3)
    s3 = pivot - (range_ * 3)
    
    # Align levels to 6h timeframe
    ema34_aligned = ema34_1d_aligned
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, pivots, volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Fade at S1/S2: price touches level and reverses upward
            # Long: touch S1 or S2, close above it, in uptrend, volume spike
            if ((low[i] <= s1_aligned[i] or low[i] <= s2_aligned[i]) and 
                close[i] > s1_aligned[i] and close[i] > s2_aligned[i] and 
                close[i] > ema_trend and vol_spike_val):
                signals[i] = size
                position = 1
            # Fade at R1/R2: price touches level and reverses downward
            # Short: touch R1 or R2, close below it, in downtrend, volume spike
            elif ((high[i] >= r1_aligned[i] or high[i] >= r2_aligned[i]) and 
                  close[i] < r1_aligned[i] and close[i] < r2_aligned[i] and 
                  close[i] < ema_trend and vol_spike_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R1/R2 (mean reversion) or trend reverses
            if high[i] >= r1_aligned[i] or high[i] >= r2_aligned[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches S1/S2 (mean reversion) or trend reverses
            if low[i] <= s1_aligned[i] or low[i] <= s2_aligned[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ClassicPivot_S1S2_R1R2_Fade_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0