#!/usr/bin/env python3
"""
1d_Donchian_20_Breakout_Volume_Trend_v1
Strategy: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation.
Long: Close breaks above 20-day high in uptrend with volume > 1.5x 20-day average.
Short: Close breaks below 20-day low in downtrend with volume > 1.5x 20-day average.
Designed for 1d timeframe: ~10-20 trades/year per symbol (40-80 total over 4 years).
Works in bull/bear via trend filter and breakout logic with volume confirmation.
"""

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
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i]
        breakout_down = close[i] < low_20[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout up
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakout down
            elif downtrend and vol_confirm and breakout_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or opposite breakout
            if not uptrend or breakout_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or opposite breakout
            if not downtrend or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_20_Breakout_Volume_Trend_v1"
timeframe = "1d"
leverage = 1.0