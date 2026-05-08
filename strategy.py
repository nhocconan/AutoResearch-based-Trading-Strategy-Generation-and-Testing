#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with weekly trend filter and volume spike
# Go long when price breaks above 10-day high with weekly EMA(34) uptrend and volume spike
# Go short when price breaks below 10-day low with weekly EMA(34) downtrend and volume spike
# Uses 1d timeframe to target 10-30 trades/year, avoiding excessive frequency
# Designed to work in both bull and bear markets by requiring trend alignment
# Volume confirmation reduces false breakouts, weekly filter ensures trend alignment

name = "1d_Donchian10_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 10-day Donchian channels (high/low of last 10 days)
    # Use rolling window on daily data, but we need to get it aligned
    # We'll calculate on 1d data then align to 1d (which is same timeframe)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(high_10[i]) or 
            np.isnan(low_10[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        upper_channel = high_10[i]
        lower_channel = low_10[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel + weekly uptrend + volume spike
            if (close[i] > upper_channel and 
                close[i] > ema34_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel + weekly downtrend + volume spike
            elif (close[i] < lower_channel and 
                  close[i] < ema34_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower channel OR weekly trend turns down
            if (close[i] < lower_channel or close[i] < ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper channel OR weekly trend turns up
            if (close[i] > upper_channel or close[i] > ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals