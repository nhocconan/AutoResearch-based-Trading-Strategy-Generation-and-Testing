#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# In bull markets: buy when price breaks above 20-day high, trend up (price > 200 EMA).
# In bear markets: sell when price breaks below 20-day low, trend down (price < 200 EMA).
# Uses 1w EMA200 as trend filter to avoid counter-trend trades.
# Volume filter ensures breakouts have participation.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_donchian20_1w_ema200_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    ema_200 = pd.Series(df_1w['close']).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align indicators to 1d
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Exit: price breaks below 20-day low or trend turns down
            if close[i] < low_20_aligned[i] or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 20-day high or trend turns up
            if close[i] > high_20_aligned[i] or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter and trend alignment
            if vol_filter:
                # Long: price breaks above 20-day high and trend up (price > EMA200)
                if close[i] > high_20_aligned[i] and close[i] > ema_200_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 20-day low and trend down (price < EMA200)
                elif close[i] < low_20_aligned[i] and close[i] < ema_200_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals