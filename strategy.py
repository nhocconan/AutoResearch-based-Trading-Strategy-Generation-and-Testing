#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_TrendFilter_v1
Hypothesis: 6h Donchian(20) breakout with weekly pivot trend filter and volume confirmation captures strong momentum moves. Uses weekly higher timeframe for trend bias (avoids counter-trend trades) and volume spike to confirm breakout strength. Designed for 6h to target 12-37 trades/year with discrete sizing (0.25). Works in both bull and bear markets by aligning with weekly trend direction.
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter (more stable than shorter periods)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Previous weekly Camarilla levels for pivot points (more robust than daily)
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    
    # Weekly Camarilla R3/S3 levels (stronger support/resistance)
    weekly_range = prev_weekly_high - prev_weekly_low
    R3_weekly = prev_weekly_close + weekly_range * 1.1 / 4
    S3_weekly = prev_weekly_close - weekly_range * 1.1 / 4
    
    # Align weekly levels to 6h timeframe
    R3_weekly_aligned = align_htf_to_ltf(prices, df_1w, R3_weekly)
    S3_weekly_aligned = align_htf_to_ltf(prices, df_1w, S3_weekly)
    
    # 6h Donchian(20) breakout levels
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 1.8 * 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of weekly EMA50 (50), Donchian (20), volume MA (24)
    start_idx = max(50, 20, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        ema_val = ema_50_1w_aligned[i]
        r3_weekly = R3_weekly_aligned[i]
        s3_weekly = S3_weekly_aligned[i]
        upper_donchian = highest_high[i]
        lower_donchian = lowest_low[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(r3_weekly) or np.isnan(s3_weekly) or 
            np.isnan(upper_donchian) or np.isnan(lower_donchian)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Weekly trend filter: price vs weekly EMA50
        weekly_uptrend = close_val > ema_val
        weekly_downtrend = close_val < ema_val
        
        # Long: price breaks above 6h Donchian upper with weekly uptrend and volume spike
        # Additional filter: breakout must also exceed weekly R3 for stronger confirmation
        long_condition = (close_val > upper_donchian) and weekly_uptrend and vol_spike and (close_val > r3_weekly)
        # Short: price breaks below 6h Donchian lower with weekly downtrend and volume spike
        # Additional filter: breakout must also break below weekly S3 for stronger confirmation
        short_condition = (close_val < lower_donchian) and weekly_downtrend and vol_spike and (close_val < s3_weekly)
        
        # Exit: price re-enters the opposite Donchian level (mean reversion within the channel)
        long_exit = (position == 1 and close_val < lower_donchian)
        short_exit = (position == -1 and close_val > upper_donchian)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0