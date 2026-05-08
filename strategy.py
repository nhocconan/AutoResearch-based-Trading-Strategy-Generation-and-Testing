#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA trend filter
# Long when price breaks above Donchian high with above-average volume and 12h EMA rising
# Short when price breaks below Donchian low with above-average volume and 12h EMA falling
# Exit when price crosses the Donchian midline (10-period average) or volume dries up
# Uses discrete position sizing (0.25) to minimize fee churn and allow multiple timeframes to align
# Targets 20-50 trades per year to stay within profitable frequency bounds

name = "4h_Donchian20_Volume_12hEMA_Trend"
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
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2.0
    
    # Volume confirmation: current volume vs 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume above average
        vol_condition = volume[i] > vol_avg[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high with volume and 12h EMA rising
            if close[i] > high_max[i] and vol_condition and ema34_12h_aligned[i] > ema34_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low with volume and 12h EMA falling
            elif close[i] < low_min[i] and vol_condition and ema34_12h_aligned[i] < ema34_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midline or volume dries up
            if close[i] < donchian_mid[i] or not vol_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midline or volume dries up
            if close[i] > donchian_mid[i] or not vol_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals