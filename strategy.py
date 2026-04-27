#!/usr/bin/env python3
"""
4h_Donchian_Breakout_1dTrend_VolumeSpike
Hypothesis: Buy when price breaks above Donchian(20) high with 1d EMA50 uptrend and volume spike; sell when breaks below Donchian low with 1d EMA50 downtrend and volume spike. Works in bull via breakouts above rising EMA50, bear via breakdowns below falling EMA50. Volume spike filters false breakouts. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Start after Donchian warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if EMA not ready
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema_50_aligned[i]
        donch_high = high_20[i]
        donch_low = low_20[i]
        vol_spike_now = vol_spike[i]
        
        if position == 0:
            # Long: break above Donchian high + uptrend + volume spike
            if close[i] > donch_high and close[i] > ema_trend and vol_spike_now:
                signals[i] = size
                position = 1
            # Short: break below Donchian low + downtrend + volume spike
            elif close[i] < donch_low and close[i] < ema_trend and vol_spike_now:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: break below Donchian low
            if close[i] < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above Donchian high
            if close[i] > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0