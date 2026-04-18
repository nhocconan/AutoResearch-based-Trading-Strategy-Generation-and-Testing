#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_TrendFilter_V1
Strategy: 4h Camarilla pivot level R1/S1 breakout with daily EMA trend filter and volume confirmation.
Long: Price breaks above R1 in daily uptrend with volume confirmation.
Short: Price breaks below S1 in daily downtrend with volume filter.
Designed for 4h timeframe: ~15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via trend filter and pivot structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume average (20-period) for volume confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_aligned[i]
        
        # Camarilla breakout conditions
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above R1
            if uptrend and vol_confirm and breakout_long:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakout below S1
            elif downtrend and vol_confirm and breakout_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or breakdown below S1 (mean reversion)
            if not uptrend or close[i] < s1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or breakout above R1 (mean reversion)
            if not downtrend or close[i] > r1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_TrendFilter_V1"
timeframe = "4h"
leverage = 1.0