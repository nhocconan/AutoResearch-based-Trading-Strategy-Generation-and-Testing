#!/usr/bin/env python3
"""
4h_donchian_20_12h_trend_volume_v1
Hypothesis: On 4-hour timeframe, use 20-period Donchian channel breakout with 12-hour trend filter and volume confirmation. 
Enter long when price breaks above upper band with 12h trend up and volume > 1.5x average, short when price breaks below lower band with 12h trend down and volume > 1.5x average.
Exit when price touches opposite band or trend reverses. Designed for low frequency (20-50 trades/year) to minimize fee drag while capturing strong breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_12h_trend_volume_v1"
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
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20 for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 20-period Donchian bands
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if 12h trend data not available
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if Donchian bands not available
        if np.isnan(highest_20[i]) or np.isnan(lowest_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches lower band or 12h trend turns down
            if close[i] <= lowest_20[i] or ema_12h_aligned[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches upper band or 12h trend turns up
            if close[i] >= highest_20[i] or ema_12h_aligned[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with 12h trend up and volume confirmation
            long_entry = (close[i] > highest_20[i]) and (ema_12h_aligned[i] > close[i]) and vol_confirm
            # Short entry: price breaks below lower band with 12h trend down and volume confirmation
            short_entry = (close[i] < lowest_20[i]) and (ema_12h_aligned[i] < close[i]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals