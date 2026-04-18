#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA12 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema12_1w = close_1w_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Align weekly EMA12 to daily
    ema12_1w_aligned = align_htf_to_ltf(prices, df_1w, ema12_1w)
    
    # Calculate daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_20 = high_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian channel
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema12_1w_aligned[i]) or np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above upper Donchian + weekly uptrend + volume
            if (close[i] > upper_20[i] and 
                ema12_1w_aligned[i] > ema12_1w_aligned[i-1] and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian + weekly downtrend + volume
            elif (close[i] < lower_20[i] and 
                  ema12_1w_aligned[i] < ema12_1w_aligned[i-1] and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price falls below lower Donchian
            if close[i] < lower_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper Donchian
            if close[i] > upper_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA12_Trend_Volume"
timeframe = "1d"
leverage = 1.0