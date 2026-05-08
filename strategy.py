#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d trend filter and volume confirmation
# Breakouts above 60-period high or below 60-period low with 1d EMA50 alignment and volume spike.
# Uses 60-period Donchian for fewer, higher-quality signals on 6h timeframe.
# Designed to capture strong trends while minimizing whipsaws in ranging markets.

name = "6h_Donchian60_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 70:
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
    
    # 1d volume spike (2x 20-period EMA)
    vol_ema_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (vol_ema_1d * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 60-period Donchian channels on 6h
    donchian_high = pd.Series(high).rolling(window=60, min_periods=60).max().values
    donchian_low = pd.Series(low).rolling(window=60, min_periods=60).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for full Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND above 1d EMA50 AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1d_aligned[i] and vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND below 1d EMA50 AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1d_aligned[i] and vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend fails
            if (close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend fails
            if (close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals