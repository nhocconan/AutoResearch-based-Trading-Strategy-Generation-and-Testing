#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly EMA(50) trend filter and volume confirmation
# Go long on breakout above upper band with weekly uptrend and volume spike,
# short on breakdown below lower band with weekly downtrend and volume spike.
# Designed for low trade frequency and robustness in both bull and bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "12h_Donchian20_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channels on 12h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume spike: current volume > 2.0 * 30-period average
    vol_ma = np.full(n, np.nan)
    for i in range(30-1, n):
        vol_ma[i] = np.mean(volume[i-30+1:i+1])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, lookback)  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: break above upper band + weekly uptrend + volume spike
            if (close[i] > upper_band and 
                close[i] > ema50_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower band + weekly downtrend + volume spike
            elif (close[i] < lower_band and 
                  close[i] < ema50_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower band
            if close[i] < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band
            if close[i] > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals