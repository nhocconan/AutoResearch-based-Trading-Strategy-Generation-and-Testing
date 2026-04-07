#!/usr/bin/env python3
"""
4h_donchian_20_12h_trend_volume_v2
Hypothesis: On 4-hour timeframe, use 20-period Donchian channel breakouts for entry, filtered by 12-hour EMA trend and volume confirmation. This strategy captures medium-term trends with institutional participation, using a simple, robust structure to avoid overtrading. Target: 20-50 trades/year per symbol to minimize fee drag while performing in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_12h_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h EMA(20) to 4h timeframe
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average on 4h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low (stop) or reaches Donchian high (take profit)
            if low[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high (stop) or reaches Donchian low (take profit)
            if high[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long breakout: price breaks above Donchian high with bullish trend
                if high[i] >= donchian_high[i] and ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price breaks below Donchian low with bearish trend
                elif low[i] <= donchian_low[i] and ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals