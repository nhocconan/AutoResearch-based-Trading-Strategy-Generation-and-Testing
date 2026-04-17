#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_VolumeSpike_v1
Breakout of Camarilla R1/S1 levels on 12h timeframe with volume spike confirmation.
Trend filter: price above/below weekly EMA200.
Exit when price returns to pivot point or volume drops below average.
Designed to capture institutional breakouts with volume confirmation in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === Calculate Pivot Point and Camarilla Levels from previous day ===
    # Use daily OHLC from previous day to calculate today's levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate pivot and camarilla levels for each day
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 12
    # S1 = Pivot - (H - L) * 1.1 / 12
    # R2 = Pivot + (H - L) * 1.1 / 6
    # S2 = Pivot - (H - L) * 1.1 / 6
    # R3 = Pivot + (H - L) * 1.1 / 4
    # S3 = Pivot - (H - L) * 1.1 / 4
    # R4 = Pivot + (H - L) * 1.1 / 2
    # S4 = Pivot - (H - L) * 1.1 / 2
    
    # We'll calculate these for each day and then align to 12h timeframe
    # For each daily bar, calculate levels that apply to the NEXT day
    # So we shift the calculated levels forward by 1 day
    
    # Previous day's OHLC for level calculation
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate pivot point from previous day
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    R1 = pivot + (range_val * 1.1 / 12)
    S1 = pivot - (range_val * 1.1 / 12)
    R2 = pivot + (range_val * 1.1 / 6)
    S2 = pivot - (range_val * 1.1 / 6)
    R3 = pivot + (range_val * 1.1 / 4)
    S3 = pivot - (range_val * 1.1 / 4)
    R4 = pivot + (range_val * 1.1 / 2)
    S4 = pivot - (range_val * 1.1 / 2)
    
    # Align daily levels to 12h timeframe
    # We need to align each level separately
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # === Volume Spike Detection ===
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Weekly EMA200 for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup period - need enough data for all indicators
    warmup = 200  # For weekly EMA200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long breakout: price breaks above R1 with volume spike, above weekly EMA200
            if (close[i] > R1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short breakdown: price breaks below S1 with volume spike, below weekly EMA200
            elif (close[i] < S1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot point OR volume drops below average
            if (close[i] <= pivot_aligned[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point OR volume drops below average
            if (close[i] >= pivot_aligned[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0