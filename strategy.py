#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
- Uses daily timeframe to determine trend (price above/below 200 SMA)
- Enters on 4h Donchian channel breakout in direction of daily trend
- Volume filter ensures breakouts have conviction
- Works in bull/bear markets because trend filter adapts to long/short bias
- Target: 15-30 trades/year (60-120 total over 4 years) to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter: 200 SMA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    trend_1d = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # 4h Donchian channel (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_window, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_1d[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches opposite Donchian band or trend fails
            if close[i] <= donchian_low[i] or close[i] < trend_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches opposite Donchian band or trend fails
            if close[i] >= donchian_high[i] or close[i] > trend_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine trend direction
            bullish_trend = close[i] > trend_1d[i]
            bearish_trend = close[i] < trend_1d[i]
            
            # Long: price breaks above Donchian high + bullish trend + volume
            if (close[i] > donchian_high[i] and 
                bullish_trend and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low + bearish trend + volume
            elif (close[i] < donchian_low[i] and 
                  bearish_trend and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals