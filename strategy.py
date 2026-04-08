#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume
# Hypothesis: 12h Donchian(20) breakout filtered by 1d EMA50 trend and volume spike.
# Long when price breaks above upper Donchian band in uptrend (price > EMA50) with volume > 1.5x average.
# Short when price breaks below lower Donchian band in downtrend (price < EMA50) with volume spike.
# Uses tight entry conditions to target ~20-40 trades/year (~80-160 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate average volume for volume spike filter (20-period SMA of volume)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = vol_sma * 1.5
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike_threshold[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below midpoint of Donchian channel OR trend turns against us
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] < midpoint) or (close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above midpoint of Donchian channel OR trend turns against us
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] > midpoint) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band with uptrend and volume spike
            if (close[i] > highest_high[i]) and (close[i] > ema_50_1d_aligned[i]) and (volume[i] > volume_spike_threshold[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian band with downtrend and volume spike
            elif (close[i] < lowest_low[i]) and (close[i] < ema_50_1d_aligned[i]) and (volume[i] > volume_spike_threshold[i]):
                position = -1
                signals[i] = -0.25
    
    return signals