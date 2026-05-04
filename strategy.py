#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume spike
# Donchian breakouts capture momentum in both bull and bear markets.
# 1w EMA50 ensures we only trade in the dominant weekly trend direction.
# Volume spike (>1.5x 20-period EMA) confirms institutional participation.
# Designed for 1d timeframe targeting 30-100 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "1d_Donchian20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper band + above weekly EMA50 + volume
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band + below weekly EMA50 + volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower band OR weekly trend turns bearish
            if (close[i] < lowest_low[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper band OR weekly trend turns bullish
            if (close[i] > highest_high[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals