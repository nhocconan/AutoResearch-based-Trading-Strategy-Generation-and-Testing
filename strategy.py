#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hTrend_Volume
Hypothesis: Camarilla pivot (R3/S3) breakout on 6h with 12h EMA trend filter and volume spike confirmation.
Trades with the trend in both bull and bear markets using 12h EMA50 as trend filter.
Targets 12-37 trades/year to minimize fee drift.
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
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot levels for current day
        # Need previous day's OHLC (1d data)
        day_idx = i // 4  # 4 = 24/6 (6h bars per day)
        if day_idx < 1:
            signals[i] = 0.0
            continue
            
        prev_day_idx = day_idx - 1
        if prev_day_idx >= len(df_1d):
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC from 1d data
        ph = df_1d['high'].iloc[prev_day_idx]
        pl = df_1d['low'].iloc[prev_day_idx]
        pc = df_1d['close'].iloc[prev_day_idx]
        
        # Camarilla levels
        range_val = ph - pl
        r3 = pc + (range_val * 1.1 / 4)
        s3 = pc - (range_val * 1.1 / 4)
        
        # Trend direction from 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: >2.0x 20-period MA
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Breakout conditions
        long_breakout = close[i] > r3
        short_breakout = close[i] < s3
        
        # Entry logic
        long_entry = vol_confirm and trend_up and long_breakout
        short_entry = vol_confirm and trend_down and short_breakout
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = (close[i] < s3) or (not trend_up)
        short_exit = (close[i] > r3) or (not trend_down)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0