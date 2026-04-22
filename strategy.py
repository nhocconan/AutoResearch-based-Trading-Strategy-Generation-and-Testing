#!/usr/bin/env python3
"""
12h Donchian Breakout with 1w Trend Filter and Volume Confirmation
Long when price breaks above Donchian upper band (20) AND 1w EMA(20) is bullish AND volume > 50-period average
Short when price breaks below Donchian lower band (20) AND 1w EMA(20) is bearish AND volume > 50-period average
Exit when price reverses to Donchian midpoint or volume dries up
Works in bull/bear by combining trend filter with breakout logic and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-week EMA(20) for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Donchian channels (20-period) - using close prices for breakouts
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume filter: 50-period average volume
    vol_series = pd.Series(volume)
    avg_volume = vol_series.rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band, 1w EMA bullish, volume above average
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band, 1w EMA bearish, volume above average
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume[i] > avg_volume[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls to middle or volume dries up
                if (close[i] < donchian_middle[i] or 
                    volume[i] < avg_volume[i] * 0.8):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises to middle or volume dries up
                if (close[i] > donchian_middle[i] or 
                    volume[i] < avg_volume[i] * 0.8):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0