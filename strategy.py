#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
# Uses 6h Donchian channels (20-period high/low) to identify breakouts.
# Long when price breaks above upper band with volume and 12h EMA up.
# Short when price breaks below lower band with volume and 12h EMA down.
# Designed to capture trends in both bull and bear markets by following 12h EMA.
# Donchian breakouts capture momentum bursts while EMA filter ensures trend alignment.
name = "6h_Donchian20_12hEMA_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band + volume + 12h EMA up
            if (price > high_20[i] and 
                vol_confirm[i] and price > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band + volume + 12h EMA down
            elif (price < low_20[i] and 
                  vol_confirm[i] and price < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below lower Donchian band or below 12h EMA
            if price < low_20[i] or price < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above upper Donchian band or above 12h EMA
            if price > high_20[i] or price > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals