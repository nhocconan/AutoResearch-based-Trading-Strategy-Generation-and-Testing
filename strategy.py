#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_20_VOLUME_CONFIRMATION_1D_TREND_FILTER
# Hypothesis: 4-hour Donchian(20) breakouts with daily EMA trend filter and volume confirmation
# capture momentum in both bull and bear markets. Volume spikes filter out false breakouts.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4H_DONCHIAN_BREAKOUT_20_VOLUME_CONFIRMATION_1D_TREND_FILTER"
timeframe = "4h"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume (20-period) for volume confirmation
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA to 4h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with volume confirmation and uptrend
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * avg_volume[i] and  # Volume spike
                close[i] > ema34_aligned[i]):        # Uptrend filter
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with volume confirmation and downtrend
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * avg_volume[i] and  # Volume spike
                  close[i] < ema34_aligned[i]):        # Downtrend filter
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below Donchian low or trend reversal
            if (close[i] < donchian_low[i] or 
                close[i] <= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above Donchian high or trend reversal
            if (close[i] > donchian_high[i] or 
                close[i] >= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals