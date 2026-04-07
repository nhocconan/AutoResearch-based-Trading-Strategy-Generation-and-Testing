#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v2
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) upper band with volume > 1.5x average and price > 1d EMA50; enter short when price breaks below Donchian(20) lower band with volume > 1.5x average and price < 1d EMA50. Uses volume confirmation and trend filter to avoid false breakouts. Designed for 20-35 trades/year to minimize fee drag while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
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
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band (20-period)
            if close[i] < low_min[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band (20-period)
            if close[i] > high_max[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if vol_ok:
                # Long: break above Donchian upper band + price > 1d EMA50
                if (high[i] > high_max[i-1] and 
                    close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.30
                # Short: break below Donchian lower band + price < 1d EMA50
                elif (low[i] < low_min[i-1] and 
                      close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.30
    
    return signals