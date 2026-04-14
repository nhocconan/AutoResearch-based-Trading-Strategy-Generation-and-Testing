#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with volume confirmation and weekly trend filter
# Donchian breakouts capture momentum in both bull and bear markets
# Volume > 1.5x average confirms breakout strength
# Weekly EMA20 trend filter ensures we trade with the higher timeframe trend
# Exit when price returns to the middle of the Donchian channel
# Target: 10-20 trades/year per symbol to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend filter
    ema_len = 20
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian Channel (20 periods)
    dc_len = 20
    upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().values
    lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().values
    middle = (upper + lower) / 2
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(middle[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above weekly EMA20 for long, below for short
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + volume + uptrend
            if (close[i] > upper[i-1] and 
                volume_confirmed and 
                uptrend):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower Donchian + volume + downtrend
            elif (close[i] < lower[i-1] and 
                  volume_confirmed and 
                  downtrend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle of Donchian channel
            if close[i] < middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle of Donchian channel
            if close[i] > middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_Breakout_Volume_Trend_v1"
timeframe = "1d"
leverage = 1.0