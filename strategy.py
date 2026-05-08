#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + weekly EMA50 trend filter + volume confirmation
# Donchian breakouts capture momentum in trending markets. Weekly EMA50 filters for primary trend direction.
# Volume confirmation ensures breakouts have institutional participation.
# Targets 8-20 trades per year (~32-80 total over 4 years) to minimize fee drift.
# Works in bull markets via breakout continuation and bear markets via short breakdowns.

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels on daily
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly close
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1w[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hh = highest_high[i]
        ll = lowest_low[i]
        ema50 = ema50_1w[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: close breaks above Donchian high, above weekly EMA50, volume confirmation
            if close[i] > hh and close[i] > ema50 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: close breaks below Donchian low, below weekly EMA50, volume confirmation
            elif close[i] < ll and close[i] < ema50 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close breaks below Donchian low or below weekly EMA50
            if close[i] < ll or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close breaks above Donchian high or above weekly EMA50
            if close[i] > hh or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals