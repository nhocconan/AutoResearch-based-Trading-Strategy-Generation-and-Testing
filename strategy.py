#!/usr/bin/env python3
# 4h_Pivot_R1_S1_Breakout_Volume_Trend
# Hypothesis: In ranging markets, price often reverses at Camarilla pivot levels (R1/S1). 
# In trending markets, breakouts of these levels with volume confirmation capture momentum.
# Uses 1d Camarilla pivots for structure, 1d volume surge for institutional confirmation,
# and 4h ADX to distinguish ranging (<25) vs trending (>25) regimes.
# In ranging: fade touches of R1/S1 (mean reversion). In trending: breakout continuation.
# Stops when price reverts to central pivot (P) or ADX drops below 20.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "4h_Pivot_R1_S1_Breakout_Volume_Trend"
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
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: P = (H+L+C)/3, Range = H-L
    P = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    R1 = P + (range_1d * 1.1 / 12)
    S1 = P - (range_1d * 1.1 / 12)
    R4 = P + (range_1d * 1.1 / 2)
    S4 = P - (range_1d * 1.1 / 2)
    
    # Align pivot levels to 4h
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Calculate 1d volume surge (volume > 2x 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_surge_1d = vol_1d > (vol_ma20_1d * 2.0)
    vol_surge_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_surge_1d.astype(float))
    
    # Calculate 4h ADX for regime detection
    period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    plus_dm = np.concatenate([[0], np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)])
    minus_dm = np.concatenate([[0], np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)])
    
    # Smooth TR, +DM, -DM
    tr_sum = np.full_like(high, np.nan)
    plus_dm_sum = np.full_like(high, np.nan)
    minus_dm_sum = np.full_like(high, np.nan)
    
    for i in range(len(high)):
        if i >= period:
            tr_sum[i] = np.nansum(tr[i-period+1:i+1])
            plus_dm_sum[i] = np.nansum(plus_dm[i-period+1:i+1])
            minus_dm_sum[i] = np.nansum(minus_dm[i-period+1:i+1])
    
    # DI+ and DI-
    plus_di = np.full_like(high, np.nan)
    minus_di = np.full_like(high, np.nan)
    dx = np.full_like(high, np.nan)
    adx = np.full_like(high, np.nan)
    
    valid = ~np.isnan(tr_sum) & (tr_sum != 0)
    plus_di[valid] = 100 * plus_dm_sum[valid] / tr_sum[valid]
    minus_di[valid] = 100 * minus_dm_sum[valid] / tr_sum[valid]
    
    dx_valid = ~np.isnan(plus_di) & ~np.isnan(minus_di) & ((plus_di + minus_di) != 0)
    dx[dx_valid] = 100 * np.abs(plus_di[dx_valid] - minus_di[dx_valid]) / (plus_di[dx_valid] + minus_di[dx_valid])
    
    # ADX: smoothed DX
    adx_sum = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i >= period:
            valid_dx = dx[i-period+1:i+1]
            valid_dx = valid_dx[~np.isnan(valid_dx)]
            if len(valid_dx) > 0:
                adx_sum[i] = np.mean(valid_dx)
    
    for i in range(len(high)):
        if i >= 2*period-1 and not np.isnan(adx_sum[i]):
            adx[i] = adx_sum[i]  # Already smoothed by mean
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 2*period-1)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(P_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_surge_1d_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Regime detection: ADX > 25 = trending, ADX < 20 = ranging
        trending = adx[i] > 25
        ranging = adx[i] < 20
        
        if position == 0:
            # Ranging market: fade R1/S1 touches
            if ranging:
                # Long near S1 with volume surge
                if low[i] <= S1_aligned[i] * 1.002 and close[i] > S1_aligned[i] and vol_surge_1d_aligned[i] > 0.5:
                    signals[i] = 0.25
                    position = 1
                # Short near R1 with volume surge
                elif high[i] >= R1_aligned[i] * 0.998 and close[i] < R1_aligned[i] and vol_surge_1d_aligned[i] > 0.5:
                    signals[i] = -0.25
                    position = -1
            # Trending market: breakout of R1/S1 with volume surge
            elif trending:
                # Long breakout above R1
                if close[i] > R1_aligned[i] and vol_surge_1d_aligned[i] > 0.5:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below S1
                elif close[i] < S1_aligned[i] and vol_surge_1d_aligned[i] > 0.5:
                    signals[i] = -0.25
                    position = -1
                
        elif position == 1:
            # Long: exit if price returns to P or breaks S4 (stop) or ADX weakens
            if close[i] < P_aligned[i] or close[i] < S4_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to P or breaks R4 (stop) or ADX weakens
            if close[i] > P_aligned[i] or close[i] > R4_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals