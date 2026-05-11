#!/usr/bin/env python3
name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
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
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1w trend: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 1.8 * 50-period SMA of volume (on 12h)
    vol_sma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume > 1.8 * vol_sma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + 1d EMA34 up + 1w EMA34 up + volume spike
            if (close[i] > high_max[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + 1d EMA34 down + 1w EMA34 down + volume spike
            elif (close[i] < low_min[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle or 1d EMA34 turns down
            if (close[i] < (high_max[i] + low_min[i]) / 2 or 
                ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle or 1d EMA34 turns up
            if (close[i] > (high_max[i] + low_min[i]) / 2 or 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals