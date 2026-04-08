#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h Trend Filter and Volume Confirmation
Hypothesis: Donchian(20) breakouts on 4h timeframe capture breakout momentum.
12h EMA(50) filter ensures alignment with higher timeframe trend.
Volume > 1.5x 20-period average confirms breakout strength.
Works in bull/bear by using trend filter to avoid counter-trend breakouts.
Target: 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # 12h OHLC for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels for each 12h bar
    # Donchian(20): Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (shifted by 1 for completed 12h bars only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower OR trend turns bearish
            if (close[i] <= donchian_lower_aligned[i] or 
                close[i] <= ema_50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper OR trend turns bullish
            if (close[i] >= donchian_upper_aligned[i] or 
                close[i] >= ema_50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian upper with uptrend and volume
            if (close[i] >= donchian_upper_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower with downtrend and volume
            elif (close[i] <= donchian_lower_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals