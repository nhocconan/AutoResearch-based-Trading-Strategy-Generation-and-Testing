#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1-week EMA50 trend filter and volume spike.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 13-period EMA).
# Long when Bull Power > 0, Bear Power < 0, price > weekly EMA50, and volume > 2x average.
# Short when Bear Power > 0, Bull Power < 0, price < weekly EMA50, and volume > 2x average.
# Uses weekly trend to avoid counter-trend trades in strong trends.
# Volume spike ensures entry during momentum expansion.
# Designed for ~15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following weekly trend and requiring bull/bear power alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on weekly close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 13-period EMA for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Bull Power > 0, Bear Power < 0, price > weekly EMA50, volume spike
        if (bull_power[i] > 0 and 
            bear_power[i] < 0 and 
            close[i] > ema50_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Bear Power > 0, Bull Power < 0, price < weekly EMA50, volume spike
        elif (bear_power[i] > 0 and 
              bull_power[i] < 0 and 
              close[i] < ema50_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0