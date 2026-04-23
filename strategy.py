#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA50 trend filter.
- Long: price breaks above 4h Donchian upper (20) + volume > 1.5x 20-period average + price > 1d EMA50
- Short: price breaks below 4h Donchian lower (20) + volume > 1.5x 20-period average + price < 1d EMA50
- Exit: price crosses 4h Donchian middle (10-period average of high/low) OR opposite signal
- Uses 4h for signal direction (structure), 1h for entry timing precision
- Session filter: 08-20 UTC to avoid low-liquidity hours
- Position size: 0.20 (20%) to control drawdown
- Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag on 1h timeframe
- Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_20_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_20_mid = (donchian_20_high + donchian_20_low) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe (completed-bar timing)
    donchian_20_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_high)
    donchian_20_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_low)
    donchian_20_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_mid)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 50)  # Need 20 for Donchian, 20 for volume MA, 50 for 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_20_high_aligned[i]) or 
            np.isnan(donchian_20_low_aligned[i]) or 
            np.isnan(donchian_20_mid_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper + volume spike + price > 1d EMA50 (uptrend)
            if volume_spike and close[i] > donchian_20_high_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower + volume spike + price < 1d EMA50 (downtrend)
            elif volume_spike and close[i] < donchian_20_low_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below 4h Donchian middle OR opposite signal
            if close[i] < donchian_20_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above 4h Donchian middle OR opposite signal
            if close[i] > donchian_20_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hBreakout_1dEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0