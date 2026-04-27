#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Daily timeframe strategy using Donchian(20) breakouts with 1-week EMA50 trend filter and volume spike confirmation. Works in bull/bear via strong trend filter (avoids counter-trend breakouts) and volume confirmation (ensures institutional participation). Targets 30-100 total trades over 4 years (7-25/year) with 0.30 position size. Uses discrete levels to minimize fee drag. Avoids chop via EMA50 trend alignment.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Donchian(20) channels from daily data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.30   # Position size (30% of capital)
    
    # Warmup: need 1w EMA50 (50), Donchian(20) (20), volume avg (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_channel = high_roll[i]
        lower_channel = low_roll[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with 1w EMA50 alignment and volume confirmation
            long_condition = (close_val > upper_channel and 
                            close_val > ema_val and 
                            vol_conf)
            short_condition = (close_val < lower_channel and 
                             close_val < ema_val and 
                             vol_conf)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower channel (20-day low)
            if close_val < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above Donchian upper channel (20-day high)
            if close_val > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0