#!/usr/bin/env python3
name = "4h_Donchian20_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume spike filter: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # 1d ATR stop filter: ATR(14) > 0.02 * close (avoid choppy markets)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d,
                     np.absolute(high_1d - np.roll(close_1d, 1)),
                     np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    atr_14_1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    volatility_filter = atr_14_1d_aligned > 0.02 * close_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50[i]) or
            np.isnan(vol_ma20_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian high + uptrend + volume + volatility
            if close[i] > high_20[i] and close[i] > ema_50[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low + downtrend + volume + volatility
            elif close[i] < low_20[i] and close[i] < ema_50[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Breakdown below Donchian low or trend down
            if close[i] < low_20[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Breakout above Donchian high or trend up
            if close[i] > high_20[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals