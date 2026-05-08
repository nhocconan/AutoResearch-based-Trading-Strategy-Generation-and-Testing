#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Uses daily trend (price > 200 EMA) to filter direction, 12h Donchian breakout for entry,
# and volume spike confirmation. Designed for low trade frequency (12-37/year) on 12h timeframe
# Works in bull markets via breakouts and bear markets via trend-filtered short opportunities

name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 200 EMA for daily trend
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 12h Donchian channels (20 periods)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema * 2.0)
    
    # Align 1d trend to 12h timeframe
    trend_1d = (close_1d > ema_200).astype(int) * 2 - 1  # 1 for uptrend, -1 for downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is invalid
        if np.isnan(trend_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high + uptrend + volume spike
            if (close[i] > donchian_high[i] and 
                trend_1d_aligned[i] == 1 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + downtrend + volume spike
            elif (close[i] < donchian_low[i] and 
                  trend_1d_aligned[i] == -1 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend changes
            if (close[i] < donchian_low[i] or trend_1d_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend changes
            if (close[i] > donchian_high[i] or trend_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals