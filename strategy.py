#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation
# Uses 12h EMA50 for HTF trend alignment to reduce whipsaw vs shorter timeframes
# Donchian channels from 6h provide clear breakout structure
# Breakout with volume spike confirms institutional participation
# 12h EMA50 trend filter ensures alignment with intermediate trend
# Works in both bull and bear markets by following 12h trend
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag
# Discrete position sizing: 0.25 (25% of capital) to minimize fee churn while maintaining reasonable exposure

name = "6h_Donchian20_Breakout_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 6h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper Donchian channel with volume spike AND price > 12h EMA50 (bullish trend)
            if (close[i] > upper_channel[i] and 
                volume_spike[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian channel with volume spike AND price < 12h EMA50 (bearish trend)
            elif (close[i] < lower_channel[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below lower Donchian channel OR below 12h EMA50 (trend change)
            if close[i] < lower_channel[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above upper Donchian channel OR above 12h EMA50 (trend change)
            if close[i] > upper_channel[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals