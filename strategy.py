#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot S1/S3 breakout with 1d volume spike and 4h EMA50 trend filter.
# Long when price breaks above S3 AND 1d volume > 2x 20-period average AND close > EMA50.
# Short when price breaks below S1 AND 1d volume > 2x 20-period average AND close < EMA50.
# Exit when price crosses back below S2 (for long) or above S4 (for short).
# Uses Camarilla pivot levels for precise entries with volume and trend confirmation.
# Target: 80-150 total trades over 4 years (20-38/year) for low fee drift.

name = "4h_Camarilla_S1S3_1dVolume_4hEMA50"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA50 trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter_4h = volume > (2.0 * vol_ma20)
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    camarilla_h5 = close_prev + 1.1 * (high_prev - low_prev) * 1.1 / 6
    camarilla_h4 = close_prev + 1.1 * (high_prev - low_prev) * 1.1 / 4
    camarilla_h3 = close_prev + 1.1 * (high_prev - low_prev) * 1.1 / 3
    camarilla_l3 = close_prev - 1.1 * (high_prev - low_prev) * 1.1 / 3
    camarilla_l4 = close_prev - 1.1 * (high_prev - low_prev) * 1.1 / 4
    camarilla_l5 = close_prev - 1.1 * (high_prev - low_prev) * 1.1 / 6
    
    # Key levels: S1 = L3, S3 = L5, R1 = H3, R3 = H5
    s1 = camarilla_l3
    s3 = camarilla_l5
    r1 = camarilla_h3
    r3 = camarilla_h5
    s2 = camarilla_l4
    s4 = camarilla_l5  # Actually L5 is S3, S4 would be below but we use S3 for exit
    r2 = camarilla_h4
    r4 = camarilla_h5  # R4 would be above but we use R3 for exit
    
    # Align 1d Camarilla levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s3)  # S3 as exit for shorts
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r3)  # R3 as exit for longs
    
    # 1d volume filter: current volume > 2x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = vol_1d > (2.0 * vol_ma20_1d)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50[i]) or np.isnan(volume_filter_4h[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(volume_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3, 1d volume spike, EMA50 uptrend
            long_cond = (close[i] > r3_aligned[i]) and volume_filter_1d_aligned[i] and (close[i] > ema50[i])
            # Short conditions: break below S3, 1d volume spike, EMA50 downtrend
            short_cond = (close[i] < s3_aligned[i]) and volume_filter_1d_aligned[i] and (close[i] < ema50[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below R2 (or S4 for safety)
            if close[i] < r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above S2 (or R4 for safety)
            if close[i] > s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals