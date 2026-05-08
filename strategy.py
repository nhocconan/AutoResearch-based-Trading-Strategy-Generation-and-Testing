#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation
# Williams %R measures overbought/oversold conditions; combined with 12h trend for direction
# Volume spike required to confirm momentum. Designed to work in both bull and bear markets
# by following higher timeframe trend and fading extremes in ranging markets.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_WilliamsR_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend direction
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_12h_val = ema34_12h_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: oversold + uptrend + volume spike
            if (wr < -80 and 
                close[i] > ema34_12h_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: overbought + downtrend + volume spike
            elif (wr > -20 and 
                  close[i] < ema34_12h_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral OR trend turns down
            if (wr > -50 or close[i] < ema34_12h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral OR trend turns up
            if (wr < -50 or close[i] > ema34_12h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals