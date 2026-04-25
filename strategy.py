#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeConfirm
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation (>1.8x 20-day mean volume). Uses weekly HTF for trend alignment to capture medium-term momentum while reducing whipsaw in both bull and bear markets. Volume confirmation ensures breakouts have conviction. Discrete position sizing (0.25) minimizes fee churn. Target: 15-25 trades/year per symbol.
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
    
    # Get weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on weekly for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA50 to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels from daily data
    # Upper band: 20-period high, Lower band: 20-period low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-day mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and volume mean
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper in uptrend (price > weekly EMA50) with volume confirmation
            # Short: price breaks below Donchian lower in downtrend (price < weekly EMA50) with volume confirmation
            long_signal = (close[i] > donchian_upper[i]) and (close[i] > ema_50_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] < donchian_lower[i]) and (close[i] < ema_50_aligned[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Donchian lower (mean reversion) or trend reversal
            exit_signal = (close[i] < donchian_lower[i]) or (close[i] < ema_50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian upper (mean reversion) or trend reversal
            exit_signal = (close[i] > donchian_upper[i]) or (close[i] > ema_50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0