#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly pivot direction + volume confirmation
    # Weekly pivot (from prior week) provides directional bias for trend alignment
    # Donchian breakout captures momentum with statistical significance
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: breaks through key levels with trend and volume confirmation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    # R2 = PP + (H - L)
    # S2 = PP - (H - L)
    # R3 = H + 2*(PP - L)
    # S3 = L - 2*(H - PP)
    pp_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pp_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pp_1w)
    
    # Use weekly pivot direction: above PP = bullish bias, below PP = bearish bias
    weekly_bias = pp_1w  # Use pivot point as reference for bias
    
    # Align weekly bias to 6h timeframe (using previous week's bias)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Donchian channel (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(weekly_bias_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume spike and price above weekly pivot (bullish bias)
            if close[i] > highest_high[i] and vol_spike[i] and close[i] > weekly_bias_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume spike and price below weekly pivot (bearish bias)
            elif close[i] < lowest_low[i] and vol_spike[i] and close[i] < weekly_bias_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level (lower band for longs, upper band for shorts)
            if position == 1:
                if close[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_20_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0