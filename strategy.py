#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Exponential Moving Average crossover with 1d trend filter and volume confirmation
# Uses EMA(21) crossing EMA(55) for momentum, confirmed by 1d EMA(34) trend and volume spike.
# Designed to capture trending moves in both bull and bear markets while avoiding whipsaws.
# Target: 60-120 total trades over 4 years (15-30/year) to stay within optimal range.

name = "4h_EMACross_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # EMA crossover components on 4h data
    ema_fast = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slow = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Volume spike: current volume > 1.5 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        fast_val = ema_fast[i]
        slow_val = ema_slow[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: EMA(21) crosses above EMA(55) + uptrend + volume spike
            if (fast_val > slow_val and 
                ema_fast[i-1] <= ema_slow[i-1] and  # crossed this bar
                close[i] > ema34_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: EMA(21) crosses below EMA(55) + downtrend + volume spike
            elif (fast_val < slow_val and 
                  ema_fast[i-1] >= ema_slow[i-1] and  # crossed this bar
                  close[i] < ema34_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: EMA(21) crosses below EMA(55) OR price breaks below trend
            if (fast_val < slow_val and ema_fast[i-1] >= ema_slow[i-1]) or close[i] < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: EMA(21) crosses above EMA(55) OR price breaks above trend
            if (fast_val > slow_val and ema_fast[i-1] <= ema_slow[i-1]) or close[i] > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals