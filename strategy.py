#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume spike confirmation
# Uses daily Donchian channels for breakout signals in the direction of weekly trend (EMA50).
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Designed for 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns.
# Weekly EMA50 filter prevents counter-trend trading during strong weekly trends.

name = "1d_Donchian20_1wEMA50_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align weekly EMA50 to daily timeframe (wait for completed weekly bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 20-period average volume for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period high/low)
        period = 20
        if i >= period:
            highest_high = np.max(high[i-period+1:i+1])
            lowest_low = np.min(low[i-period+1:i+1])
        else:
            # Not enough data for Donchian calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average volume
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian band + volume spike + price above weekly EMA50 (uptrend)
            if (close[i] > highest_high and 
                volume_spike and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian band + volume spike + price below weekly EMA50 (downtrend)
            elif (close[i] < lowest_low and 
                  volume_spike and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR breaks below lower band
            if close[i] <= highest_high and close[i] >= lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR breaks above upper band
            if close[i] <= highest_high and close[i] >= lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals