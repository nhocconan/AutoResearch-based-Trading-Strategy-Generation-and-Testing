#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike
Hypothesis: Use 4h timeframe with Donchian(20) breakout, confirmed by 12h EMA50 trend and volume spike.
Long when: price breaks above upper Donchian + 12h EMA50 uptrend + volume > 1.5 * avg volume.
Short when: price breaks below lower Donchian + 12h EMA50 downtrend + volume > 1.5 * avg volume.
Exit when: price reverts to middle Donchian (20-period mean) or opposite Donchian band touched.
Uses discrete 0.25 position size to limit fee drag. Designed for BTC/ETH:
- Donchian breakouts capture strong momentum moves
- 12h EMA50 filter ensures trading with the higher timeframe trend
- Volume confirmation avoids low-validity breakouts
- Targets 20-50 trades/year for optimal test generalization.
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
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low).rolling(window=20, min_periods=20)
    upper = high_roll.max().values
    lower = low_roll.min().values
    middle = ((upper + lower) / 2).values  # 20-period mean
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian/volume, 50 for 12h EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above upper Donchian + 12h EMA50 uptrend + volume spike
            long_entry = (close_val > upper[i]) and \
                       (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below lower Donchian + 12h EMA50 downtrend + volume spike
            short_entry = (close_val < lower[i]) and \
                        (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]) and \
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
            # Long - exit when price reverts to middle or touches lower band
            if (close_val < middle[i]) or (close_val < lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to middle or touches upper band
            if (close_val > middle[i]) or (close_val > upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0