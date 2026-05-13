#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d volume spike confirmation and 1w EMA50 trend filter.
# Long when price breaks above 12h Donchian upper channel AND 1d volume > 2.0 * 20-period average volume AND price > 1w EMA50.
# Short when price breaks below 12h Donchian lower channel AND 1d volume > 2.0 * 20-period average volume AND price < 1w EMA50.
# Exit when price reverts to the 12h Donchian midpoint (mean of upper and lower channel).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing breakouts with volume confirmation in the prevailing weekly trend.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Donchian20_VolumeSpike_1wEMA50_TrendFilter_v1"
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
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel AND volume spike AND price > 1w EMA50
            if (close[i] > highest_high[i] and 
                volume_spike_aligned[i] > 0.5 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower channel AND volume spike AND price < 1w EMA50
            elif (close[i] < lowest_low[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals