#!/usr/bin/env python3
"""
1h_4h_1d_trend_following_ma_volume_v1
Hypothesis: Use 4h moving average for trend direction and 1d moving average for trend strength filter.
Enter long when 1h price is above both 4h MA and 1d MA with volume confirmation.
Enter short when 1h price is below both 4h MA and 1d MA with volume confirmation.
Exit when price crosses back below/above the 4h MA or volume drops.
Designed to work in both bull and bear markets by following the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_trend_following_ma_volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA(30) for trend direction
    close_4h = df_4h['close'].values
    ema_30_4h = pd.Series(close_4h).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Get 1d data for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend strength
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 1h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Align HTF indicators to 1h timeframe
    ema_30_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_30_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_30_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below 4h EMA OR volume drops
            if close[i] < ema_30_4h_aligned[i] or not volume_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price crosses above 4h EMA OR volume drops
            if close[i] > ema_30_4h_aligned[i] or not volume_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Position size
        else:  # Flat, look for entry
            # Long entry: Price above both 4h EMA and 1d EMA + volume
            if (close[i] > ema_30_4h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: Price below both 4h EMA and 1d EMA + volume
            elif (close[i] < ema_30_4h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals