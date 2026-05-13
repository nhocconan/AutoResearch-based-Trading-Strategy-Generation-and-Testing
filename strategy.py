#!/usr/bin/env python3
# 4h_Pivot_Reversal_Scalp
# Hypothesis: At 4h timeframe, price often reverses at daily pivot points (S1/S2/R1/R2) with volume confirmation.
# Works in both bull and bear markets as reversals occur at key levels regardless of trend.
# Uses 1d Camarilla pivots + volume spike + tight stop via reversal signal.
# Low frequency due to requiring exact pivot proximity and volume confirmation.

name = "4h_Pivot_Reversal_Scalp"
timeframe = "4h"
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

    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    # R2 = C + (H - L) * 1.1/6
    # S2 = C - (H - L) * 1.1/6
    # Using previous day's values to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    pp = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    range_1d = np.roll(high_1d, 1) - np.roll(low_1d, 1)
    r1 = np.roll(close_1d, 1) + range_1d * 1.1 / 12
    s1 = np.roll(close_1d, 1) - range_1d * 1.1 / 12
    r2 = np.roll(close_1d, 1) + range_1d * 1.1 / 6
    s2 = np.roll(close_1d, 1) - range_1d * 1.1 / 6
    
    # Align pivot levels to 4h timeframe (will use previous day's pivots)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume spike: volume > 1.8 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Proximity to pivot levels (within 0.15% of price)
        price = close[i]
        near_s1 = abs(price - s1_aligned[i]) / price < 0.0015
        near_s2 = abs(price - s2_aligned[i]) / price < 0.0015
        near_r1 = abs(price - r1_aligned[i]) / price < 0.0015
        near_r2 = abs(price - r2_aligned[i]) / price < 0.0015
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price near S1 or S2 with volume spike (bounce off support)
            if (near_s1 or near_s2) and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price near R1 or R2 with volume spike (rejection at resistance)
            elif (near_r1 or near_r2) and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot point or shows rejection
            if price >= pp_aligned[i] or near_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot point or shows bounce
            if price <= pp_aligned[i] or near_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals