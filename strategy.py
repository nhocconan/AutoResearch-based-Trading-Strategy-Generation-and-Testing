#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses weekly Camarilla pivot levels for structural bias (R3/S3 from prior week)
# Donchian(20) breakout confirms momentum in direction of weekly bias
# Volume spike (>1.5x 20-bar average) confirms breakout strength
# Discrete sizing 0.25 to limit fee drag; target 50-150 total trades over 4 years (12-37/year)
# Works in bull markets via breakout continuation and bear markets via fade at weekly extremes

name = "6h_Donchian20_WeeklyCamarilla_R3S3_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels (using previous weekly bar)
    # Camarilla: R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    camarilla_high_1w = []
    camarilla_low_1w = []
    for i in range(len(close_1w)):
        if i == 0:
            camarilla_high_1w.append(np.nan)
            camarilla_low_1w.append(np.nan)
        else:
            h = high_1w[i-1]
            l = low_1w[i-1]
            c = close_1w[i-1]
            r3 = c + ((h - l) * 1.1 / 4)
            s3 = c - ((h - l) * 1.1 / 4)
            camarilla_high_1w.append(r3)
            camarilla_low_1w.append(s3)
    
    camarilla_high_1w = np.array(camarilla_high_1w)
    camarilla_low_1w = np.array(camarilla_low_1w)
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_high_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_high_1w)
    camarilla_low_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_low_1w)
    
    # Calculate Donchian(20) channels on 6h data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_high_1w_aligned[i]) or np.isnan(camarilla_low_1w_aligned[i]) or 
            np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high AND above weekly R3 AND volume spike
            if close[i] > high_ma_20[i] and close[i] > camarilla_high_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Donchian low AND below weekly S3 AND volume spike
            elif close[i] < low_ma_20[i] and close[i] < camarilla_low_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests weekly S3 or Donchian low
            if close[i] <= camarilla_low_1w_aligned[i] or close[i] <= low_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests weekly R3 or Donchian high
            if close[i] >= camarilla_high_1w_aligned[i] or close[i] >= high_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals