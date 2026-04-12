#!/usr/bin/env python3
"""
6h_1d_1w_Camarilla_Pivot_Breakout_Trend_v1
Hypothesis: Uses daily and weekly pivot levels for breakout entries with volume confirmation and 6h EMA trend filter.
Focus on high-probability breakouts in trending markets, targeting 12-30 trades/year per symbol.
Designed to work in both bull and bear markets by filtering entries with trend and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Camarilla_Pivot_Breakout_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 6h EMA50 for trend filter
    ema_50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla levels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d)
    
    # Calculate weekly Camarilla levels
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    camarilla_r3_1w = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_s3_1w = close_1w - 1.1 * (high_1w - low_1w)
    
    # Align Camarilla levels to 6h timeframe (wait for close of daily/weekly bar)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_6h[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(camarilla_r3_1w_aligned[i]) or np.isnan(camarilla_s3_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Trend filter: price above/below 6h EMA50
        uptrend = close[i] > ema_50_6h[i]
        downtrend = close[i] < ema_50_6h[i]
        
        # Breakout conditions: price must break both daily and weekly levels
        breakout_up = close[i] > camarilla_r3_1d_aligned[i] and close[i] > camarilla_r3_1w_aligned[i]
        breakdown_down = close[i] < camarilla_s3_1d_aligned[i] and close[i] < camarilla_s3_1w_aligned[i]
        
        # Entry conditions: only trade in direction of 6h trend
        long_entry = breakout_up and volume_filter and uptrend
        short_entry = breakdown_down and volume_filter and downtrend
        
        # Exit conditions: return to opposite level or trend reversal
        long_exit = (close[i] < camarilla_s3_1d_aligned[i] and close[i] < camarilla_s3_1w_aligned[i]) or (not uptrend)
        short_exit = (close[i] > camarilla_r3_1d_aligned[i] and close[i] > camarilla_r3_1w_aligned[i]) or (not downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals