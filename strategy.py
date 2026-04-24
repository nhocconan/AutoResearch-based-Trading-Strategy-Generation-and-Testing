#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Donchian(20) on 4h chart identifies key breakout levels with clear structure.
- 12h EMA50 provides higher-timeframe trend filter to align with momentum and reduce counter-trend trades.
- Volume spike (>2.0x 24-period average) confirms breakout validity and reduces false signals.
- Discrete position sizing (0.25) minimizes fee churn while allowing meaningful returns.
- Target trades: 75-200 total over 4 years (19-50/year) on 4h timeframe to avoid fee drag.
- Works in bull/bear markets via 12h trend filter and volatility-based volume confirmation.
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
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper (20-period high) and lower (20-period low)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (using previous completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 2.0x 24-period average volume (4h * 6 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike and above 12h EMA50 (bullish higher-timeframe trend)
            if close[i] > donchian_high_aligned[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and below 12h EMA50 (bearish higher-timeframe trend)
            elif close[i] < donchian_low_aligned[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low OR below 12h EMA50 (trend change)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high OR above 12h EMA50 (trend change)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0