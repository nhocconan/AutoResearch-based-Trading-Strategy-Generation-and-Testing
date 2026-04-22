#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter
    # Weekly pivot points provide institutional support/resistance levels
    # Breakouts in direction of weekly trend have higher follow-through
    # Works in both bull/bear markets by filtering false breakouts
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points using standard formula
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    H_weekly = df_1w['high'].values
    L_weekly = df_1w['low'].values
    C_weekly = df_1w['close'].values
    
    pivot = (H_weekly + L_weekly + C_weekly) / 3
    R1 = 2 * pivot - L_weekly
    S1 = 2 * pivot - H_weekly
    R2 = pivot + (H_weekly - L_weekly)
    S2 = pivot - (H_weekly - L_weekly)
    R3 = H_weekly + 2 * (pivot - L_weekly)
    S3 = L_weekly - 2 * (H_weekly - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Calculate Donchian channels (20-period) on 6h
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.8 * vol_ma20  # Require strong volume for breakout
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(S2_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high AND above weekly R1
            # Strong bullish signal when breaking above weekly resistance
            if high[i] > high_max20[i] and close[i] > R1_aligned[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Donchian low AND below weekly S1
            # Strong bearish signal when breaking below weekly support
            elif low[i] < low_min20[i] and close[i] < S1_aligned[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to weekly pivot level or opposite Donchian break
            if position == 1:
                if close[i] < pivot_aligned[i] or low[i] < low_min20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_aligned[i] or high[i] > high_max20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_20_WeeklyPivot_R1S1_Breakout_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0