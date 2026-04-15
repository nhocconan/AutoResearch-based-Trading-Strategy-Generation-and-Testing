#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 12h EMA Filter and Volume Spike
# Uses 4-hour Donchian channel breakouts (20-period) for trend capture,
# filtered by 12-hour EMA(50) to ensure alignment with medium-term trend,
# and confirmed by volume spikes (2x median volume). Works in bull markets
# (breakouts above upper band) and bear markets (breakouts below lower band).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h Donchian channels (20-period)
    # Use rolling window on high/low for Donchian bands
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if EMA data not available
        if np.isnan(ema_12h_aligned[i]):
            continue
            
        # Long entry: price breaks above Donchian high + above 12h EMA + volume spike
        if (close[i] > donchian_high[i] and
            close[i] > ema_12h_aligned[i] and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + below 12h EMA + volume spike
        elif (close[i] < donchian_low[i] and
              close[i] < ema_12h_aligned[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Donchian breakout or price crosses 12h EMA in opposite direction
        elif position == 1 and (close[i] < donchian_low[i] or close[i] < ema_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high[i] or close[i] > ema_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_EMA_Volume_Spike"
timeframe = "4h"
leverage = 1.0