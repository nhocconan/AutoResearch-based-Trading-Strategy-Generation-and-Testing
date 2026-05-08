#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume average for volume spike filter
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 12h Donchian channel (20 periods)
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donch_high[i] = np.max(high[start_idx:i+1])
        donch_low[i] = np.min(low[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, daily uptrend, volume spike
            long_cond = (close[i] > donch_high[i] and 
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume[i] > (1.5 * vol_ma20_1d_aligned[i]))
            
            # Short: price breaks below Donchian low, daily downtrend, volume spike
            short_cond = (close[i] < donch_low[i] and 
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume[i] > (1.5 * vol_ma20_1d_aligned[i]))
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian low
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian high
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals