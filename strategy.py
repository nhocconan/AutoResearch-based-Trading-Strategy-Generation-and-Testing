#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Trade 6h Donchian(20) breakouts in the direction of weekly trend with volume confirmation.
Weekly trend filter avoids counter-trend trades during 2022 bear and 2025 sideways markets.
Volume confirmation ensures breakout validity. Position size 0.25 limits drawdown.
Targets 12-37 trades/year (50-150 over 4 years) by requiring confluence of trend, breakout, and volume.
Works in bull markets (buy breakouts with uptrend) and bear markets (sell breakdowns with downtrend).
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
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h Donchian(20) channels
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly HTF trend (bullish = price above weekly EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian upper + weekly uptrend + volume confirmation
            long_setup = (close[i] > upper[i]) and htf_1w_bullish and volume_confirm
            
            # Short setup: price breaks below Donchian lower + weekly downtrend + volume confirmation
            short_setup = (close[i] < lower[i]) and htf_1w_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Donchian lower (stop) OR weekly trend turns bearish
            if (close[i] <= lower[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian upper (stop) OR weekly trend turns bullish
            if (close[i] >= upper[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0