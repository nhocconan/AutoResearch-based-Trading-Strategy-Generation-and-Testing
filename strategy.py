#!/usr/bin/env python3
"""
6h_WeeklyPivot_PB_Donchian20_1dEMA50
Hypothesis: On 6h timeframe, enter long when price pulls back to weekly pivot (PP) after breaking above Donchian(20) high AND 1d trend is up (close > EMA50). Enter short when price pulls back to weekly pivot after breaking below Donchian(20) low AND 1d trend is down (close < EMA50). Uses volume confirmation (volume > 1.5x 20-period average) to filter false breakouts. Exit on Donchian(10) opposite break or trend reversal. Target: 12-37 trades/year. Works in bull (breakouts) and bear (pullbacks in trends).
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    # Weekly pivot point (PP) = (H + L + C) / 3
    pp = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    
    # Donchian channels for breakout detection (20-period)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian channels for exit (10-period)
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50, Donchian(20), Donchian(10), volume MA
    start_idx = max(50, 20, 10)  # EMA50 needs 50, others need 20/10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout detection
        broke_above_dc20 = close[i] > donchian_high_20[i]
        broke_below_dc20 = close[i] < donchian_low_20[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: broke above DC20, now pulling back to weekly PP, volume spike, uptrend
            long_signal = broke_above_dc20 and abs(close[i] - pp_aligned[i]) < (donchian_high_20[i] - donchian_low_20[i]) * 0.05 and volume_spike[i] and trend_uptrend
            
            # Short: broke below DC20, now pulling back to weekly PP, volume spike, downtrend
            short_signal = broke_below_dc20 and abs(close[i] - pp_aligned[i]) < (donchian_high_20[i] - donchian_low_20[i]) * 0.05 and volume_spike[i] and trend_downtrend
            
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
            # Exit: Donchian(10) breakdown OR trend change to downtrend
            if close[i] < donchian_low_10[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Donchian(10) breakout OR trend change to uptrend
            if close[i] > donchian_high_10[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_PB_Donchian20_1dEMA50"
timeframe = "6h"
leverage = 1.0