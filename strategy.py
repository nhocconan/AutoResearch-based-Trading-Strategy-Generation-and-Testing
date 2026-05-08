#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA(50) trend filter and volume spike
# Long when price breaks above 20-day high with weekly uptrend and volume spike
# Short when price breaks below 20-day low with weekly downtrend and volume spike
# Uses daily timeframe targeting 10-25 trades/year to minimize fee drag
# Volume confirmation ensures momentum behind breakouts
# Trend filter avoids counter-trend trades in choppy markets

name = "1d_Donchian20_WeeklyTrend_Volume"
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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-day) on daily data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: current volume > 2.0 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel + weekly uptrend + volume spike
            if (close[i] > upper_channel and 
                close[i] > ema50_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel + weekly downtrend + volume spike
            elif (close[i] < lower_channel and 
                  close[i] < ema50_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower channel OR weekly trend turns down
            if (close[i] < lower_channel or close[i] < ema50_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper channel OR weekly trend turns up
            if (close[i] > upper_channel or close[i] > ema50_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals