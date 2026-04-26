#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakouts in the direction of weekly trend with volume confirmation.
Long when: price breaks above daily Donchian(20) high + weekly close > weekly EMA(50) + volume > 1.5x avg volume(20).
Short when: price breaks below daily Donchian(20) low + weekly close < weekly EMA(50) + volume > 1.5x avg volume(20).
Exit when: price reverts to daily Donchian midpoint or weekly trend flips.
Designed for BTC/ETH: captures strong momentum moves in both bull and bear markets while avoiding chop.
Target: 15-25 trades/year for low fee drag and strong test generalization.
"""

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
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily Donchian(20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian/volume, 50 for weekly EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for breakout in direction of weekly trend with volume spike
            if vol_spike[i]:
                # Long: break above Donchian high + weekly close above EMA(50)
                long_entry = (close_val > donch_high[i]) and (weekly_close[-1] > ema_50[-1]) if len(weekly_close) > 0 else False
                # Short: break below Donchian low + weekly close below EMA(50)
                short_entry = (close_val < donch_low[i]) and (weekly_close[-1] < ema_50[-1]) if len(weekly_close) > 0 else False
                
                if long_entry:
                    signals[i] = size
                    position = 1
                elif short_entry:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to midpoint or weekly trend flips down
            if close_val < donch_mid[i] or (len(weekly_close) > 0 and weekly_close[-1] < ema_50[-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to midpoint or weekly trend flips up
            if close_val > donch_mid[i] or (len(weekly_close) > 0 and weekly_close[-1] > ema_50[-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0