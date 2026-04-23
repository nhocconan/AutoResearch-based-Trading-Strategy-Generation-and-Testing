#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel breakout with 1w EMA50 trend filter and volume confirmation.
- Long: Close breaks above Donchian(20) high + price > 1w EMA50 (uptrend) + volume > 2x 20-period avg
- Short: Close breaks below Donchian(20) low + price < 1w EMA50 (downtrend) + volume > 2x 20-period avg
- Exit: Opposite Donchian breakout OR close crosses 1w EMA50 (trend reversal)
- Uses 1d timeframe with 1w HTF for trend alignment to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag on 1d timeframe
- Works in bull markets (trend continuation via breakouts) and bear markets (avoids counter-trend trades via HTF filter)
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
    
    # Volume confirmation: > 2x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 for 1w trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume MA, 50 for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Donchian high + price > 1w EMA50 (uptrend) + volume spike
            if volume_spike and close[i] > donchian_high[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian low + price < 1w EMA50 (downtrend) + volume spike
            elif volume_spike and close[i] < donchian_low[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Donchian low OR price < 1w EMA50 (trend break)
            if close[i] < donchian_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Donchian high OR price > 1w EMA50 (trend break)
            if close[i] > donchian_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0