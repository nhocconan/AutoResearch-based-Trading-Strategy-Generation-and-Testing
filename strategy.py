#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1-week EMA trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions: values below -80 = oversold, above -20 = overbought.
# In trending markets: Buy when %R crosses above -80 from below, Sell when %R crosses below -20 from above.
# Uses 1-week EMA for trend filter to avoid counter-trend trades and capture major trends.
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
    
    # 14-period Williams %R calculation
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 34-period EMA on 1-week close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trending market logic with Williams %R and volume filter
        if close[i] > ema34_1w_aligned[i] and volume_filter[i]:  # Uptrend
            # Buy when Williams %R crosses above -80 from below
            if williams_r[i] > -80 and (i == start_idx or williams_r[i-1] <= -80):
                signals[i] = 0.25
                position = 1
            # Exit long when Williams %R rises above -20 (overbought)
            elif position == 1 and williams_r[i] >= -20:
                signals[i] = 0.0
                position = 0
        elif close[i] < ema34_1w_aligned[i] and volume_filter[i]:  # Downtrend
            # Sell when Williams %R crosses below -20 from above
            if williams_r[i] < -20 and (i == start_idx or williams_r[i-1] >= -20):
                signals[i] = -0.25
                position = -1
            # Exit short when Williams %R falls below -80 (oversold)
            elif position == -1 and williams_r[i] <= -80:
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

name = "12h_WilliamsR_1wEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0