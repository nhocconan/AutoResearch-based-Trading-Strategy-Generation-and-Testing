#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h trend filter + volume confirmation
# Uses Donchian(20) channels on 4h for breakout signals, filtered by 12h EMA trend
# and volume confirmation to avoid false breakouts. Works in trending markets
# (both bull and bear) by only taking breakouts in the direction of the 12h trend.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_12h_aligned[i]):
            continue
        
        # Long entry: price breaks above Donchian high + volume confirmation + price above 12h EMA
        if (close[i] > donchian_high[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            close[i] > ema_12h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume confirmation + price below 12h EMA
        elif (close[i] < donchian_low[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              close[i] < ema_12h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or price crosses 12h EMA in opposite direction
        elif position == 1 and (close[i] < donchian_low[i] or close[i] < ema_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high[i] or close[i] > ema_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0