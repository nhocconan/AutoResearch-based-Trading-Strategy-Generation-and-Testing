#!/usr/bin/env python3
name = "1h_4h1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # 4h close for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Previous 4h bar (completed) for pivot calculation
    # Since we need the completed 4h bar, we use index i-1 in 4h terms
    # But we'll handle this via alignment later
    
    # Daily trend: EMA 50 on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels using previous completed 4h bar
        # We need the 4h bar that closed before the current 1h bar
        # Find the index of the most recent completed 4h bar
        # 4h bars are every 16th 1h bar (since 4h = 4 * 1h)
        if i < 16:
            # Not enough data for previous 4h bar
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Index of the most recent completed 4h bar in 4h data
        # Current 1h bar index i corresponds to 4h bar index i//16
        # But we want the COMPLETED 4h bar, so we use (i//16) - 1 if we're not at a 4h boundary
        # Actually, simpler: use the 4h bar at index (i//16) - 1 for the previous completed 4h bar
        # Except when i is exactly at a 4h boundary, then the previous 4h bar is (i//16) - 1
        four_h_bar_index = i // 16
        if four_h_bar_index < 1:
            # Not enough 4h history
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Get the previous completed 4h bar data
        prev_4h_idx = four_h_bar_index - 1
        if prev_4h_idx < 0 or prev_4h_idx >= len(close_4h):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Previous completed 4h bar OHLC
        ph = high_4h[prev_4h_idx]
        pl = low_4h[prev_4h_idx]
        pc = close_4h[prev_4h_idx]
        
        # Calculate Camarilla levels
        range_ = ph - pl
        if range_ <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Camarilla R3, S3, R4, S4
        R3 = pc + range_ * 1.1 / 4
        S3 = pc - range_ * 1.1 / 4
        R4 = pc + range_ * 1.1 / 2
        S4 = pc - range_ * 1.1 / 2
        
        if position == 0:
            # Long: price breaks above R3 with volume and session filter, in uptrend
            if close[i] > R3 and volume_filter[i] and session_filter[i] and trend_up[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with volume and session filter, in downtrend
            elif close[i] < S3 and volume_filter[i] and session_filter[i] and not trend_up[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes
            if close[i] < S3 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes
            if close[i] > R3 or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals