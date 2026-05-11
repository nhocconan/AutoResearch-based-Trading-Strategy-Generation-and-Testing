#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for Camarilla levels, trend filter, and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous day's Camarilla levels
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_r3_1d = close_1d + range_1d * 1.1 / 4
    camarilla_s3_1d = close_1d - range_1d * 1.1 / 4
    camarilla_r4_1d = close_1d + range_1d * 1.1 / 2
    camarilla_s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    camarilla_r3_1d = np.roll(camarilla_r3_1d, 1)
    camarilla_s3_1d = np.roll(camarilla_s3_1d, 1)
    camarilla_r4_1d = np.roll(camarilla_r4_1d, 1)
    camarilla_s4_1d = np.roll(camarilla_s4_1d, 1)
    camarilla_r3_1d[0] = np.nan
    camarilla_s3_1d[0] = np.nan
    camarilla_r4_1d[0] = np.nan
    camarilla_s4_1d[0] = np.nan
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume spike detection (volume > 1.5 * 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (vol_ma_1d * 1.5)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 35  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and uptrend (price > EMA34)
            if (close[i] > r3_aligned[i] and 
                vol_spike_1d_aligned[i] > 0.5 and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and downtrend (price < EMA34)
            elif (close[i] < s3_aligned[i] and 
                  vol_spike_1d_aligned[i] > 0.5 and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (reversal) or touches R4 (take profit)
            if (close[i] < s3_aligned[i] or 
                close[i] > r4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (reversal) or touches S4 (take profit)
            if (close[i] > r3_aligned[i] or 
                close[i] < s4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals