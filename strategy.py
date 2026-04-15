#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h EMA filter and volume confirmation
# - Long when price breaks above 6h Donchian(20) high + price > 12h EMA(50) + volume spike
# - Short when price breaks below 6h Donchian(20) low + price < 12h EMA(50) + volume spike
# - Exit when price crosses 12h EMA(50) in opposite direction
# - Uses EMA as trend filter to avoid counter-trend breakouts
# - Volume confirmation ensures breakout conviction
# - Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data for Donchian channels (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Load 12h data for EMA filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 6h
    # Highest high over last 20 periods
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lowest low over last 20 periods
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA(50) on 12h
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 6h indicators to lower timeframe (assuming 6h is primary, but we need alignment for safety)
    # Since we're using 6h as primary, we need to align Donchian to our actual resolution
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Align 12h EMA to lower timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_12h_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + above 12h EMA + volume spike
        if (close[i] > donchian_high_aligned[i] and
            close[i] > ema_12h_aligned[i] and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + below 12h EMA + volume spike
        elif (close[i] < donchian_low_aligned[i] and
              close[i] < ema_12h_aligned[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price crosses 12h EMA in opposite direction
        elif position == 1 and close[i] < ema_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_12hEMA_Volume"
timeframe = "6h"
leverage = 1.0