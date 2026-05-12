#!/usr/bin/env python3
# 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_AND_VOLUME
# Hypothesis: Combining Camarilla R3/S3 breakouts with 1d trend (EMA34) and volume spike (2x average)
# creates a high-conviction signal that works in both bull and bear markets. Volume confirmation
# reduces false breakouts, while the trend filter ensures alignment with higher-timeframe momentum.
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years).

name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_AND_VOLUME"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation, trend filter, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # R3 = C + (H-L)*1.25/2, S3 = C - (H-L)*1.25/2
    r3 = close_1d + (high_1d - low_1d) * 1.25 / 2
    s3 = close_1d - (high_1d - low_1d) * 1.25 / 2
    
    # EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 20-period average volume for volume spike detection
    vol_ma20 = pd.Series(volume_1d := df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_aligned[i]) or np.isnan(vol_ma20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike condition: current 12h volume > 2x 20-period average
        volume_spike = volume[i] > 2 * vol_ma20_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R3 in uptrend with volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_aligned[i] and
                volume_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 in downtrend with volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_aligned[i] and
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S3 or trend reversal
            if (close[i] < s3_aligned[i] or 
                close[i] <= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R3 or trend reversal
            if (close[i] > r3_aligned[i] or 
                close[i] >= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals