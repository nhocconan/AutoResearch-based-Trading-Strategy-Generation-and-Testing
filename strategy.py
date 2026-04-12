#!/usr/bin/env python3
"""
6h_1d_wr_10_reversion
Hypothesis: 6-hour strategy using Williams %R overbought/oversold levels (10/90) for mean reversion, filtered by 1-day trend to avoid counter-trend trades in strong trends. Works in both bull and bear markets by only taking mean-reversion trades aligned with the higher timeframe trend.
Target: 50-150 total trades over 4 years = 12-37/year to minimize fee drag.
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
    
    # Get daily data for trend and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 14-period Williams %R on daily: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    wr_1d = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Daily EMA50 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    wr_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6-period Williams %R on 6h for entry timing
    highest_high_6h = pd.Series(high).rolling(window=6, min_periods=6).max().values
    lowest_low_6h = pd.Series(low).rolling(window=6, min_periods=6).min().values
    wr_6h = (highest_high_6h - close) / (highest_high_6h - lowest_low_6h) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(wr_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(wr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1-day trend: price above EMA50 = uptrend, below = downtrend
        uptrend = close_1d[i] > ema50_1d[i] if i < len(close_1d) else ema50_1d_aligned[i] > 0  # fallback
        # Use aligned EMA for trend at current 6h bar
        trend_up = ema50_1d_aligned[i] > 0  # placeholder, will use price comparison below
        # Since we don't have daily close aligned, use price vs EMA as trend proxy
        # Better: use the fact that EMA50_1d_aligned represents the trend level
        # We'll determine trend by comparing current price to EMA50 from prior day
        # For simplicity, use EMA slope: upward if current EMA > past EMA
        if i >= 15:
            ema_now = ema50_1d_aligned[i]
            ema_past = ema50_1d_aligned[i-1]
            trend_up = ema_now > ema_past
            trend_down = ema_now < ema_past
        else:
            trend_up = True
            trend_down = False
        
        # Mean reversion logic: 
        # In uptrend, look for oversold (WR < -90) to go long
        # In downtrend, look for overbought (WR > -10) to go short
        if trend_up and wr_6h[i] < -90 and position != 1:
            # Long entry: oversold in uptrend
            position = 1
            signals[i] = 0.25
        elif trend_down and wr_6h[i] > -10 and position != -1:
            # Short entry: overbought in downtrend
            position = -1
            signals[i] = -0.25
        # Exit conditions: WR crosses back to neutral territory (-50)
        elif position == 1 and wr_6h[i] > -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and wr_6h[i] < -50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_wr_10_reversion"
timeframe = "6h"
leverage = 1.0