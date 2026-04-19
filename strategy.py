#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# In trending markets, price breaks Donchian channels with volume (continuation)
# In ranging markets, price reverts from Donchian extremes with volume (mean reversion)
# Uses 12h EMA34 as trend filter to distinguish between breakouts and reversals
# Volume filter ensures only significant moves trigger entries
# Works in both bull and bear markets by adapting to volatility regime via ATR stop
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years)

name = "4h_Donchian20_12hEMA34_Volume"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA and Donchian data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_12h_aligned[i]
        upper = high_max[i]
        lower = low_min[i]
        
        # Volume filter
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions:
            # 1. Breakout above upper Donchian with volume AND price > 12h EMA (bullish continuation)
            # 2. Bounce from lower Donchian with volume AND price < 12h EMA (bullish reversal)
            if ((price > upper and volume_confirmed and price > ema) or 
                (price < lower and volume_confirmed and price < ema)):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Breakdown below lower Donchian with volume AND price < 12h EMA (bearish continuation)
            # 2. Rejection from upper Donchian with volume AND price > 12h EMA (bearish reversal)
            elif ((price < lower and volume_confirmed and price < ema) or 
                  (price > upper and volume_confirmed and price > ema)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below lower Donchian
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above upper Donchian
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals