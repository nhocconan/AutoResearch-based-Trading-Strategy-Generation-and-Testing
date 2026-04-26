#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_4hEMA21_Trend_VolumeSpike
Hypothesis: Donchian(20) breakout on 4h timeframe with 4h EMA21 trend filter and volume spike confirmation (>2.0x 20-period MA). 
Trades in direction of 4h trend to avoid whipsaws. Uses discrete position sizing (0.25) to minimize fee churn.
Designed to work in both bull and bear markets by following the 4h trend, which adapts to regime changes.
Target: 20-50 trades/year (75-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels and EMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 4h EMA21 trend filter
    ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    uptrend_4h = close > ema_21_4h_aligned
    downtrend_4h = close < ema_21_4h_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 21 for EMA + 20 for Donchian/volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_21_4h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: close breaks above Donchian upper, with 4h uptrend and volume spike
            if (close[i] > donchian_upper_aligned[i] and uptrend_4h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below Donchian lower, with 4h downtrend and volume spike
            elif (close[i] < donchian_lower_aligned[i] and downtrend_4h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close drops below Donchian lower OR 4h trend changes to downtrend
            if (close[i] < donchian_lower_aligned[i] or not uptrend_4h[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close rises above Donchian upper OR 4h trend changes to uptrend
            if (close[i] > donchian_upper_aligned[i] or not downtrend_4h[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_4hEMA21_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0