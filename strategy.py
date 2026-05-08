#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot S1/R1 breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above R1 AND 1d ADX > 25 (trending) AND 12h volume > 1.5x 20-period average.
# Short when price breaks below S1 AND 1d ADX > 25 AND 12h volume > 1.5x 20-period average.
# Exit when price crosses back below/above pivot point (CP).
# Uses Camarilla pivots from daily timeframe for institutional levels with ADX filter to avoid ranging.
# Target: 80-120 total trades over 4 years (20-30/year) for low fee drift.

name = "12h_Camarilla_R1S1_1dADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for Camarilla pivot calculation and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    cp = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = cp + (range_1d * 1.1 / 12)
    s1 = cp - (range_1d * 1.1 / 12)
    r2 = cp + (range_1d * 1.1 / 6)
    s2 = cp - (range_1d * 1.1 / 6)
    
    # Calculate ADX (14-period) on 1d data
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx[np.isnan(dx)] = 0
    
    # Align 1d data to 12h timeframe
    cp_aligned = align_htf_to_ltf(prices, df_1d, cp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(cp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R1, ADX > 25, volume spike
            long_cond = (close[i] > r1_aligned[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: break below S1, ADX > 25, volume spike
            short_cond = (close[i] < s1_aligned[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below pivot point
            if close[i] < cp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above pivot point
            if close[i] > cp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals