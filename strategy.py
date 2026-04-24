#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Donchian(20) on 12h chart identifies key breakout levels with clear structure.
- 1d EMA34 provides higher-timeframe trend filter to align with momentum and reduce counter-trend trades.
- Volume spike (>2.0x 48-period average) confirms breakout validity and reduces false signals.
- Discrete position sizing (0.25) minimizes fee churn while allowing meaningful returns.
- Target trades: 50-150 total over 4 years (12-37/year) on 12h timeframe to avoid fee drag.
- Works in bull/bear markets via 1d trend filter and volatility-based volume confirmation.
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
    
    # Get 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper (20-period high) and lower (20-period low)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (using previous completed 12h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 48-period average volume (12h * 4 = 2 days)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(48, 20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike and above 1d EMA34 (bullish higher-timeframe trend)
            if close[i] > donchian_high_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and below 1d EMA34 (bearish higher-timeframe trend)
            elif close[i] < donchian_low_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low OR below 1d EMA34 (trend change)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high OR above 1d EMA34 (trend change)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0