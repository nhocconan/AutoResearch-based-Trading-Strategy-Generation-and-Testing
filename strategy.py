#!/usr/bin/env python3
"""
1d_Donchian20_VolumeSpike_HTFTrend_v1
Hypothesis: 1d Donchian(20) breakout with 1w HTF trend filter and volume confirmation. 
Long when price breaks above 20-day high with volume > 1.5x 20-day average and 1w trend up.
Short when price breaks below 20-day low with volume > 1.5x 20-day average and 1w trend down.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency 
(7-25/year) to overcome fee drag and work in both bull (breakouts) and bear (breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 20-period Donchian channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 20-period average volume for volume spike filter
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().values
    volume_spike_threshold = 1.5  # volume must be > 1.5x average
    
    # Calculate 1w EMA50 for HTF trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    htf_trend = np.where(close > ema_50_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian and volume avg)
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(htf_trend[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Check for volume spike
        volume_spike = volume[i] > (avg_volume[i] * volume_spike_threshold)
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # break above previous 20-day high
        breakout_down = close[i] < lowest_low[i-1]  # break below previous 20-day low
        
        # Exit conditions: reverse signal or Donchian middle line (optional)
        exit_long = close[i] < (highest_high[i-1] + lowest_low[i-1]) / 2  # mid-point
        exit_short = close[i] > (highest_high[i-1] + lowest_low[i-1]) / 2  # mid-point
        
        if htf_trend[i] == 1:  # 1w uptrend - look for longs
            if breakout_up and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif position == 1 and exit_long:
                signals[i] = 0.0
                position = 0
            elif position == 0:
                signals[i] = 0.0
            else:  # position == -1, flip to long
                signals[i] = 0.25
                position = 1
                
        elif htf_trend[i] == -1:  # 1w downtrend - look for shorts
            if breakout_down and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            elif position == -1 and exit_short:
                signals[i] = 0.0
                position = 0
            elif position == 0:
                signals[i] = 0.0
            else:  # position == 1, flip to short
                signals[i] = -0.25
                position = -1
    
    return signals

name = "1d_Donchian20_VolumeSpike_HTFTrend_v1"
timeframe = "1d"
leverage = 1.0