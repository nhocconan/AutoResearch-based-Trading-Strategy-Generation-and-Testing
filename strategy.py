#!/usr/bin/env python3
# Hypothesis: 12h Donchian breakout with 1w EMA200 trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high with 1w EMA200 uptrend and volume > 1.8x average
# Short when price breaks below 20-period Donchian low with 1w EMA200 downtrend and volume > 1.8x average
# Exit when price crosses the 1w EMA200 in the opposite direction
# Donchian channels capture breakouts, EMA200 filters trend direction, volume confirms momentum
# Designed for low-frequency, high-conviction trades on 12h timeframe suitable for trending and ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.28

name = "12h_Donchian_Breakout_1wEMA200_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_confirm = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, EMA200 uptrend, volume spike
            if (close[i] > donchian_high[i] and
                ema200_1w_aligned[i] > ema200_1w_aligned[i-1] and
                vol_confirm[i]):
                signals[i] = 0.28
                position = 1
            # Enter short: price breaks below Donchian low, EMA200 downtrend, volume spike
            elif (close[i] < donchian_low[i] and
                  ema200_1w_aligned[i] < ema200_1w_aligned[i-1] and
                  vol_confirm[i]):
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1w EMA200
            if close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Exit short: price crosses above 1w EMA200
            if close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals