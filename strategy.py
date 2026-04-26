#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_HTFTrend_VolumeRegime_v2
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper + 1d EMA50 uptrend + volume > 1.8 * 20-period avg.
Short when price breaks below Donchian lower + 1d EMA50 downtrend + volume > 1.8 * 20-period avg.
Exit when price crosses Donchian midpoint or opposite Donchian level touched.
Uses discrete 0.25 position size to limit fee drag. Designed for BTC/ETH:
- Donchian(20) provides clear structure and breakout signals
- 1d EMA50 filter ensures alignment with daily trend, reducing counter-trend trades
- Volume spike confirms breakout validity
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
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    midpoint = (upper + lower) / 2
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 1.8 * 20-period average (stricter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 20 for volume avg, 50 for 1d EMA
    start_idx = max(lookback, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(midpoint[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above upper + 1d EMA50 uptrend + volume spike
            long_entry = (close_val > upper[i]) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below lower + 1d EMA50 downtrend + volume spike
            short_entry = (close_val < lower[i]) and \
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
            # Long - exit when price crosses midpoint or touches lower (contrarian exit)
            if (close_val < midpoint[i]) or (close_val < lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price crosses midpoint or touches upper (contrarian exit)
            if (close_val > midpoint[i]) or (close_val > upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_HTFTrend_VolumeRegime_v2"
timeframe = "4h"
leverage = 1.0