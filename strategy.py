#!/usr/bin/env python3
"""
6h_MultiTimeframe_Breakout_Retest_V1
Hypothesis: Breakouts from 6h Donchian(20) confirmed by weekly trend and retested with volume.
Works in bull/bear by using weekly trend filter and requiring volume confirmation on breakout and retest.
Target: 20-40 trades/year per symbol with strict entry conditions.
"""
name = "6h_MultiTimeframe_Breakout_Retest_V1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for breakout context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily Donchian(20) for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_high = 0.0
    breakout_low = 0.0
    in_breakout_setup = False
    
    start_idx = 50  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for new breakout setup
            if not in_breakout_setup:
                if close[i] > donchian_high_20_aligned[i] and volume_filter[i]:
                    # Bullish breakout
                    in_breakout_setup = True
                    breakout_high = donchian_high_20_aligned[i]
                    breakout_low = None
                elif close[i] < donchian_low_20_aligned[i] and volume_filter[i]:
                    # Bearish breakout
                    in_breakout_setup = True
                    breakout_high = None
                    breakout_low = donchian_low_20_aligned[i]
            
            # Enter on retest of breakout level with volume
            if in_breakout_setup:
                if breakout_high is not None and close[i] <= breakout_high * 1.005 and close[i] >= breakout_high * 0.995:
                    if volume_filter[i] and close[i] > ema_20_1w_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                        in_breakout_setup = False  # Reset after entry
                elif breakout_low is not None and close[i] >= breakout_low * 0.995 and close[i] <= breakout_low * 1.005:
                    if volume_filter[i] and close[i] < ema_20_1w_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                        in_breakout_setup = False  # Reset after entry
            
            # Reset breakout setup if price moves too far without entry
            if in_breakout_setup:
                if breakout_high is not None and close[i] > breakout_high * 1.02:
                    in_breakout_setup = False
                elif breakout_low is not None and close[i] < breakout_low * 0.98:
                    in_breakout_setup = False
        else:
            # Exit conditions: weekly trend reversal or opposite Donchian break
            if position == 1:
                if close[i] < ema_20_1w_aligned[i] or close[i] < donchian_low_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema_20_1w_aligned[i] or close[i] > donchian_high_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals