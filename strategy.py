#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Uses Donchian(20) on 4h, 12h EMA34 for trend, volume > 1.5x 20-period average.
# Works in both bull and bear markets by filtering breakouts by higher timeframe trend.
# Target: 80-180 total trades over 4 years (20-45/year).
name = "4h_Donchian20_12hEMA34_VolumeTrend"
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
    
    # Get 12h data for EMA34 (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channels on 4h (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]):
            signals[i] = 0.0
            continue
            
        # Determine trend from 12h EMA34
        trend_up = ema_12h_aligned[i] > ema_12h_aligned[i-1]
        trend_down = ema_12h_aligned[i] < ema_12h_aligned[i-1]
        
        if position == 0:
            # Long when price breaks above Donchian high with uptrend and volume
            if close[i] > high_max[i] and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low with downtrend and volume
            elif close[i] < low_min[i] and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below Donchian low
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above Donchian high
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals