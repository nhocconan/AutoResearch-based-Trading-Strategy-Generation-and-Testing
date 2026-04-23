#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Long: Close > Donchian Upper(20) AND price > 1w EMA50 AND volume > 2.0x 24-period avg
- Short: Close < Donchian Lower(20) AND price < 1w EMA50 AND volume > 2.0x 24-period avg
- Exit: Opposite Donchian breakout OR price crosses 1w EMA50
- Uses 1w HTF for EMA50 and Donchian levels (calculated from prior completed 1w bar)
- Designed for low trade frequency (7-25/year) to minimize fee drag
- Works in bull (buy breakouts above upper channel) and bear (sell breakdowns below lower channel)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian levels from prior 1w bar (HTF = 1w)
    # Donchian: Upper = max(high, period), Lower = min(low, period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (use prior completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 24)  # Need 50 for EMA, 20 for Donchian, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs prior levels)
        breakout_up = close[i] > donchian_upper_aligned[i-1]  # Close above prior upper channel
        breakout_down = close[i] < donchian_lower_aligned[i-1]  # Close below prior lower channel
        
        if position == 0:
            # Long: Donchian upper breakout up AND price > 1w EMA50 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower breakout down AND price < 1w EMA50 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian lower breakout down OR price < 1w EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian upper breakout up OR price > 1w EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0