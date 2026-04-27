#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d trend filter and volume confirmation
# Uses daily Camarilla levels (R1, R2, S1, S2) from 1d data. Enters long at S1 bounce in uptrend,
# short at R1 resistance in downtrend. Requires volume > 1.3x 20-period average.
# Designed for 12-30 trades/year per symbol to minimize fee drag. Works in both bull and bear
# markets by following 1d EMA34 trend and fading extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H-L range from previous day
    range_1d = high_1d - low_1d
    
    # Camarilla levels (based on previous day's range)
    # R4 = close + range * 1.1/2, R3 = close + range * 1.1/4, R2 = close + range * 1.1/6, R1 = close + range * 1.1/12
    # S1 = close - range * 1.1/12, S2 = close - range * 1.1/6, S3 = close - range * 1.1/4, S4 = close - range * 1.1/2
    camarilla_r1 = close_1d + range_1d * 1.1 / 12
    camarilla_r2 = close_1d + range_1d * 1.1 / 6
    camarilla_s1 = close_1d - range_1d * 1.1 / 12
    camarilla_s2 = close_1d - range_1d * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # 34-period EMA on daily close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price at S1 support in uptrend with volume
        if (close[i] <= camarilla_s1_aligned[i] * 1.005 and  # Allow small buffer for touching
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price at R1 resistance in downtrend with volume
        elif (close[i] >= camarilla_r1_aligned[i] * 0.995 and  # Allow small buffer for touching
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0