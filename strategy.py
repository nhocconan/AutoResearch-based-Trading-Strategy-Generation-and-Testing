#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyTrend_v1
Hypothesis: Trade 6h Donchian(20) breakouts with weekly EMA(20) trend filter and volume confirmation.
Donchian breakouts capture momentum; weekly EMA ensures alignment with higher-timeframe trend.
Volume confirmation filters weak breakouts. Targets 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
Works in bull markets (breakout with weekly uptrend) and bear markets (breakdown with weekly downtrend).
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate EMA(20) on weekly for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Donchian(20) channels: 20-period high/low
    # Using pandas rolling with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 24-period average (4d average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly EMA(20), Donchian(20), volume MA(24)
    start_idx = max(20, 20, 24) + 1  # +1 for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        weekly_uptrend = close_val > ema_20_1w_aligned[i]
        weekly_downtrend = close_val < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirm AND weekly uptrend
            long_signal = (close_val > donchian_high[i]) and vol_conf and weekly_uptrend
            
            # Short: price breaks below Donchian low AND volume confirm AND weekly downtrend
            short_signal = (close_val < donchian_low[i]) and vol_conf and weekly_downtrend
            
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
            # Exit: price drops below Donchian low (failed breakout) OR weekly trend flips down
            if (close_val < donchian_low[i]) or (not weekly_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high (failed breakdown) OR weekly trend flips up
            if (close_val > donchian_high[i]) or (not weekly_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0