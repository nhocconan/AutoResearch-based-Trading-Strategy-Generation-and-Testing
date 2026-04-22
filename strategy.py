#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout (20) with weekly pivot direction filter and volume confirmation.
# Weekly pivot levels (from prior week) define support/resistance zones.
# Only take longs when price breaks above weekly pivot resistance (R1) and is above weekly pivot point.
# Only take shorts when price breaks below weekly pivot support (S1) and is below weekly pivot point.
# Volume confirmation requires current volume > 1.8x 50-period average to filter weak breakouts.
# Designed to capture strong trending moves in both bull and bear markets by aligning with weekly structure.
# Targets 15-30 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points and support/resistance levels
    # Pivot Point = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L
    r1_1w = 2 * pp_1w - low_1w
    # S1 = 2*P - H
    s1_1w = 2 * pp_1w - high_1w
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate 60-period Donchian channel on 6h data
    high = prices['high'].values
    low = prices['low'].values
    
    highest_high_60 = pd.Series(high).rolling(window=60, min_periods=60).max().values
    lowest_low_60 = pd.Series(low).rolling(window=60, min_periods=60).min().values
    
    # Calculate 50-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(pp_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(highest_high_60[i]) or 
            np.isnan(lowest_low_60[i]) or 
            np.isnan(vol_ma_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_50[i]
        hh_60 = highest_high_60[i]
        ll_60 = lowest_low_60[i]
        pp_val = pp_1w_aligned[i]
        r1_val = r1_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        
        # Volume filter: current volume > 1.8 * 50-period average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: break above Donchian high AND above weekly R1 with volume spike
            if price > hh_60 and price > r1_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low AND below weekly S1 with volume spike
            elif price < ll_60 and price < s1_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or Donchian middle reversion
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low or falls below weekly pivot
                if price < ll_60 or price < pp_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high or rises above weekly pivot
                if price > hh_60 or price > pp_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0