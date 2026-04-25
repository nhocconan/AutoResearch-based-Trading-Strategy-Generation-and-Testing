#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_Filter_v1
Hypothesis: Trade daily Donchian(20) breakouts with 1-week EMA50 trend filter. 
In bull markets (price > weekly EMA50): buy when price breaks above 20-day high. 
In bear markets (price < weekly EMA50): sell when price breaks below 20-day low. 
Requires volume > 1.3x 20-day average for confirmation to reduce false breakouts. 
Exit on opposite Donchian level touch or trend reversal. 
Position size: 0.25 to limit drawdown. 
Target: 30-100 total trades over 4 years = 7-25/year. 
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate weekly EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day average volume for confirmation
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above weekly EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-day average
        volume_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price breaks above 20-day high + 1w uptrend + volume confirmation
            long_setup = (close[i] > high_20[i]) and htf_1w_bullish and volume_confirm
            
            # Short setup: price breaks below 20-day low + 1w downtrend + volume confirmation
            short_setup = (close[i] < low_20[i]) and htf_1w_bearish and volume_confirm
            
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
            # Exit: price touches 20-day low (stop) OR 1w trend turns bearish
            if (close[i] <= low_20[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches 20-day high (stop) OR 1w trend turns bullish
            if (close[i] >= high_20[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Filter_v1"
timeframe = "1d"
leverage = 1.0