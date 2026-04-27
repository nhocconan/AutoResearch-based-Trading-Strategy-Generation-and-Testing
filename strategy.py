#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume spike.
# Elder Ray measures bull/bear power as price relative to EMA: Bull = High - EMA, Bear = Low - EMA.
# In trending markets: Buy when Bull Power > 0 and rising, Sell when Bear Power < 0 and falling.
# Uses 1d EMA for trend filter to avoid counter-trend trades.
# Volume spike confirms institutional participation.
# Target: 12-37 trades/year per symbol (50-150 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6-period EMA for Elder Ray calculation
    ema6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = high - ema6
    bear_power = low - ema6
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trending market logic with Elder Ray and volume filter
        if close[i] > ema34_1d_aligned[i] and volume_filter[i]:  # Uptrend
            # Buy when Bull Power is positive and rising (or less negative)
            if bull_power[i] > 0 and (i == start_idx or bull_power[i] > bull_power[i-1]):
                signals[i] = 0.25
                position = 1
            # Exit long when Bull Power turns negative
            elif position == 1 and bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
        elif close[i] < ema34_1d_aligned[i] and volume_filter[i]:  # Downtrend
            # Sell when Bear Power is negative and falling (or more negative)
            if bear_power[i] < 0 and (i == start_idx or bear_power[i] < bear_power[i-1]):
                signals[i] = -0.25
                position = -1
            # Exit short when Bear Power turns positive
            elif position == -1 and bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
        else:
            # Hold current position or stay flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0