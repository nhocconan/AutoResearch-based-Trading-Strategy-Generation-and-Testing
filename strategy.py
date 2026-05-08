#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# Price channel breakouts capture breakout moves in both bull and bear markets.
# 1d EMA50 filters for higher timeframe trend alignment to avoid counter-trend trades.
# 1d volume spike confirms institutional participation, reducing false breakouts.
# Designed for low-frequency, high-conviction trades to minimize fee drag on 12h timeframe.

name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume spike (2x 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (vol_ma_1d * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Donchian(20) channels on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian channels to form
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian upper band AND price above 1d EMA50 AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema50_1d_aligned[i] and vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower band AND price below 1d EMA50 AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema50_1d_aligned[i] and vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian middle OR trend fails
            middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < middle or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian middle OR trend fails
            middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > middle or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals