#!/usr/bin/env python3
name = "4H_Donchian20_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12-period EMA for trend filter
    ema12_12h = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Align 12h EMA12 to 4h timeframe
    ema12_12h_aligned = align_htf_to_ltf(prices, df_12h, ema12_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for 20-period high/low
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema12_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate 20-period high and low for Donchian channel
        period_high = np.max(high[i-20:i])
        period_low = np.min(low[i-20:i])
        
        # Determine trend
        uptrend = close[i] > ema12_12h_aligned[i]
        downtrend = close[i] < ema12_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average volume
        avg_volume = np.mean(volume[i-20:i])
        volume_confirm = volume[i] > avg_volume * 1.5
        
        if position == 0:
            # Enter long: price breaks above 20-period high + uptrend + volume confirmation
            if close[i] > period_high and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-period low + downtrend + volume confirmation
            elif close[i] < period_low and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 10-period low
            exit_low = np.min(low[i-10:i])
            if close[i] < exit_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 10-period high
            exit_high = np.max(high[i-10:i])
            if close[i] > exit_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals