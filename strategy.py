#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Donchian channel breakouts capture trend continuation and volatility expansion
# 1d EMA50 trend filter ensures alignment with higher timeframe direction
# Volume > 1.3x average confirms breakout strength and reduces false signals
# Exit when price returns to Donchian middle (mean reversion within trend)
# Target: 20-30 trades/year per symbol to avoid fee drag while maintaining edge

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_len = 50
    ema_1d = pd.Series(close_1d).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(middle[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + uptrend + volume
            if (close[i] > upper[i-1] and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower Donchian + downtrend + volume
            elif (close[i] < lower[i-1] and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle Donchian (mean reversion)
            if close[i] < middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle Donchian (mean reversion)
            if close[i] > middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_1dEMA_Volume_v1"
timeframe = "4h"
leverage = 1.0