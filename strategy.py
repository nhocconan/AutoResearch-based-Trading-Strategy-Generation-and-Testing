#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation
# Uses 20-period Donchian channels for breakout signals
# 12h EMA (50) filters trades to follow higher timeframe trend
# Volume > 1.5x 20-period average confirms breakout strength
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag
# Works in bull markets by capturing breakouts, in bear by avoiding counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA (50) for trend direction
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume confirmation and uptrend filter
            if (close[i] > highest_high[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_12h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian band with volume confirmation and downtrend filter
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_12h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian band or trend changes
            if close[i] < lowest_low[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian band or trend changes
            if close[i] > highest_high[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0