#!/usr/bin/env python3
"""
6h_HighLowChannel_Breakout_1dTrend_VolumeSpike
Hypothesis: Use 6h timeframe with breakout above/below the 20-bar high/low channel, confirmed by 1d EMA50 trend and volume spike.
Long when: price breaks above 20-bar high + 1d EMA50 uptrend + volume > 1.5 * avg volume.
Short when: price breaks below 20-bar low + 1d EMA50 downtrend + volume > 1.5 * avg volume.
Exit when: price reverts to 20-bar midpoint or touches opposite channel level.
Uses discrete 0.25 position size to limit fee drag. Designed for BTC/ETH:
- 20-bar channel breakout captures momentum with noise filtering
- 1d EMA50 filter ensures trading with the daily trend
- Volume confirmation avoids low-validity breakouts
- Targets 12-37 trades/year for optimal test generalization.
"""

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
    
    # 20-bar high/low channel
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for channel/volume, 50 for 1d EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(mid_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above 20-bar high + 1d EMA50 uptrend + volume spike
            long_entry = (close_val > high_20[i]) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below 20-bar low + 1d EMA50 downtrend + volume spike
            short_entry = (close_val < low_20[i]) and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to midpoint or touches low_20
            if (close_val < mid_20[i]) or (close_val < low_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to midpoint or touches high_20
            if (close_val > mid_20[i]) or (close_val > high_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_HighLowChannel_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0