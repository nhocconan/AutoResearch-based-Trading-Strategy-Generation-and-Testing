#!/usr/bin/env python3
# 1D_DONCHIAN20_WEEKLY_TREND_FILTER
# Hypothesis: Donchian breakout on 1d with weekly trend filter and volume confirmation.
# In weekly uptrend, go long when price breaks above 20-day high; in weekly downtrend, go short when price breaks below 20-day low.
# Weekly trend filter avoids counter-trend trades, Donchian breakout captures momentum.
# Target: 10-20 trades/year on 1d timeframe.

name = "1D_DONCHIAN20_WEEKLY_TREND_FILTER"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    weekly_ema20 = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Daily Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(weekly_ema20_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Weekly uptrend + price breaks above 20-day high + volume confirmation
            if (close[i] > weekly_ema20_aligned[i] and 
                close[i] > highest_high[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + price breaks below 20-day low + volume confirmation
            elif (close[i] < weekly_ema20_aligned[i] and 
                  close[i] < lowest_low[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-day low or weekly trend reversal
            if (close[i] < lowest_low[i] or 
                close[i] < weekly_ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-day high or weekly trend reversal
            if (close[i] > highest_high[i] or 
                close[i] > weekly_ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals