#!/usr/bin/env python3
name = "6h_Aroon_Trend_1wFilter_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    # Weekly Aroon indicator (25-period) for trend strength
    # Aroon Up = ((25 - days since 25-period high) / 25) * 100
    # Aroon Down = ((25 - days since 25-period low) / 25) * 100
    high_25 = pd.Series(df_1w['high']).rolling(window=25, min_periods=25).apply(lambda x: np.argmax(x), raw=True)
    low_25 = pd.Series(df_1w['low']).rolling(window=25, min_periods=25).apply(lambda x: np.argmin(x), raw=True)
    aroon_up = ((24 - high_25) / 24) * 100
    aroon_down = ((24 - low_25) / 24) * 100
    # Handle NaN from insufficient data
    aroon_up = aroon_up.fillna(0).values
    aroon_down = aroon_down.fillna(0).values
    
    # Aroon Oscillator: Aroon Up - Aroon Down
    aroon_osc = aroon_up - aroon_down
    
    # Align Aroon oscillator to 6h timeframe
    aroon_osc_6h = align_htf_to_ltf(prices, df_1w, aroon_osc)
    
    # Volume spike detection (1.5x 20-period average on 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(aroon_osc_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: strong uptrend (Aroon Oscillator > 50) with volume
            if aroon_osc_6h[i] > 50 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend (Aroon Oscillator < -50) with volume
            elif aroon_osc_6h[i] < -50 and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend weakening (Aroon Oscillator < 0) or volume fade
            if aroon_osc_6h[i] < 0 or volume[i] < vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend weakening (Aroon Oscillator > 0) or volume fade
            if aroon_osc_6h[i] > 0 or volume[i] < vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Aroon oscillator on weekly timeframe detects strong trends, 
# with volume confirmation on 6h for entry timing. Aroon >50 indicates strong uptrend,
# Aroon <-50 indicates strong downtrend. Works in bull (catch uptrends) and bear 
# (catch downtrends). Volume filter reduces false signals. Target 50-150 trades over 4 years.