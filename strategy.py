#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_Volume_Spike_v1
Hypothesis: Use 4h Donchian(20) breakout with 12h trend filter (EMA50) and volume spike confirmation.
Long when price breaks above upper Donchian(20) AND 12h EMA50 up AND volume > 2x 20-bar average.
Short when price breaks below lower Donchian(20) AND 12h EMA50 down AND volume > 2x 20-bar average.
Exit when price retouches the Donchian midpoint (mean reversion) or trend flips.
Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with discrete position sizing (0.25) to minimize fee drag.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue) via 12h trend filter.
Volume spike ensures momentum confirmation, reducing false breakouts.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) on 4h (primary timeframe)
    # We need at least 20 periods for Donchian
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2.0
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Donchian(20), 12h EMA(50), volume MA(20)
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(high_max[i]) or
            np.isnan(low_min[i]) or
            np.isnan(donchian_mid[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]  # 12h EMA50 rising
        trend_down = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]  # 12h EMA50 falling
        
        if position == 0:
            # Long: price breaks above upper Donchian AND volume spike AND 12h uptrend
            long_signal = (close_val > high_max[i]) and vol_spike and trend_up
            
            # Short: price breaks below lower Donchian AND volume spike AND 12h downtrend
            short_signal = (close_val < low_min[i]) and vol_spike and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price retouches Donchian midpoint OR 12h trend flips down
            if (close_val <= donchian_mid[i]) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price retouches Donchian midpoint OR 12h trend flips up
            if (close_val >= donchian_mid[i]) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0