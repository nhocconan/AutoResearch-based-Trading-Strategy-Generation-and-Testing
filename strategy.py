#!/usr/bin/env python3
name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
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
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (volume > 2x 20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma20
    
    # 1-day EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend = close > ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_spike[i]) or np.isnan(uptrend[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + uptrend + volume spike
            if close[i] > highest_high[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + downtrend + volume spike
            elif close[i] < lowest_low[i] and not uptrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below lower Donchian or trend reverses
            if close[i] < lowest_low[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above upper Donchian or trend reverses
            if close[i] > highest_high[i] or uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals