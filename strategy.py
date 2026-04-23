#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
- Long: Close > Donchian Upper(20) AND price > 1d EMA50 AND volume > 2.0x 20-period avg
- Short: Close < Donchian Lower(20) AND price < 1d EMA50 AND volume > 2.0x 20-period avg
- Exit: Opposite Donchian breakout OR price crosses 1d EMA50
- Uses 1d HTF for EMA50 and Donchian levels (calculated from prior completed bars)
- Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe
- Donchian provides clear structure, volume confirmation filters low-conviction moves
- Works in both bull (breakouts with trend) and bear (mean reversion via exits) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels from prior 1d bar (HTF = 1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian(20): upper = max(high,20), lower = min(low,20)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (use prior completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # Need 50 for EMA, 20 for Donchian, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs prior levels)
        breakout_up = close[i] > donchian_upper_aligned[i-1]  # Close above prior upper
        breakout_down = close[i] < donchian_lower_aligned[i-1]  # Close below prior lower
        
        if position == 0:
            # Long: Donchian upper breakout up AND price > 1d EMA50 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower breakout down AND price < 1d EMA50 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian lower breakout down OR price < 1d EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian upper breakout up OR price > 1d EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0