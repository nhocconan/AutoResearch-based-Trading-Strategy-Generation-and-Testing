#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter
# Donchian breakouts capture momentum; volume confirms institutional participation
# 1d EMA50 filter ensures alignment with daily trend to avoid counter-trend trades
# Designed for low trade frequency (target 20-40/year) with clear entry/exit rules
# Works in bull markets via breakouts and in bear markets via short breakdowns

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            continue
        
        # Long entry: price breaks above Donchian upper band + volume + above 1d EMA50
        if (close[i] > highest_high[i] and volume_ok[i] and 
            close[i] > ema50_1d_aligned[i] and position <= 0):
            position = 1
            signals[i] = position_size
        # Short entry: price breaks below Donchian lower band + volume + below 1d EMA50
        elif (close[i] < lowest_low[i] and volume_ok[i] and 
              close[i] < ema50_1d_aligned[i] and position >= 0):
            position = -1
            signals[i] = -position_size
        # Exit: price returns to middle of Donchian channel
        elif position == 1 and close[i] < (highest_high[i] + lowest_low[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (highest_high[i] + lowest_low[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Volume_1dEMA50_Filter"
timeframe = "4h"
leverage = 1.0