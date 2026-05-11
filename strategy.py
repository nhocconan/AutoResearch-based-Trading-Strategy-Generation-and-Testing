#!/usr/bin/env python3
"""
12h_WeeklyDonchian_20_Breakout_1dTrend_Volume
Hypothesis: Weekly Donchian(20) breakouts provide strong directional signals in both bull and bear markets.
Daily EMA50 filters for trend alignment to avoid counter-trend trades. Volume spike confirms institutional interest.
Target: 12-37 trades/year (50-150 total over 4 years) with low turnover to minimize fee drag.
Works in bull markets (buy breakouts above weekly high) and bear markets (sell breakdowns below weekly low).
"""

name = "12h_WeeklyDonchian_20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load weekly data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Load daily data ONCE for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: 20-period EMA for spike detection (using 12h volume)
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Fixed position size to minimize churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        breakout_long = close[i] > high_20_aligned[i]
        breakout_short = close[i] < low_20_aligned[i]
        price_above_ema1d = close[i] > ema50_1d_aligned[i]
        price_below_ema1d = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + above daily EMA50 + volume spike
            if breakout_long and price_above_ema1d and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below weekly Donchian low + below daily EMA50 + volume spike
            elif breakout_short and price_below_ema1d and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price breaks below weekly Donchian low OR trend reverses (close below daily EMA)
                if close[i] < low_20_aligned[i] or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks above weekly Donchian high OR trend reverses (close above daily EMA)
                if close[i] > high_20_aligned[i] or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals