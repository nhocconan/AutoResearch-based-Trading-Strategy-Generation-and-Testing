#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 12h Trend Filter + Volume Confirmation
# Long when price breaks above 4h Donchian high (20) and 12h EMA50 is rising
# Short when price breaks below 4h Donchian low (20) and 12h EMA50 is falling
# Volume must exceed 1.5x 20-bar median for confirmation
# Designed for low trade frequency and strong trend capture in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_slope_12h = np.diff(ema_50_12h_aligned, prepend=ema_50_12h_aligned[0])
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_slope_12h[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above Donchian high, 12h EMA rising, volume spike
        if (close[i] > donchian_high[i] and 
            ema_slope_12h[i] > 0 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.30
        
        # Short: Price breaks below Donchian low, 12h EMA falling, volume spike
        elif (close[i] < donchian_low[i] and 
              ema_slope_12h[i] < 0 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.30
        
        # Exit: Price returns to Donchian mid-point or EMA slope changes
        elif (i > 0 and 
              ((signals[i-1] == 0.30 and (close[i] < (donchian_high[i] + donchian_low[i]) / 2 or ema_slope_12h[i] <= 0)) or
               (signals[i-1] == -0.30 and (close[i] > (donchian_high[i] + donchian_low[i]) / 2 or ema_slope_12h[i] >= 0)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0