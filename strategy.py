# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend
Hypothesis: Fade at Camarilla R3/S3 levels on 6h with 1d EMA34 trend filter.
In bull markets, buy dips to S3; in bear markets, sell rallies to R3.
Uses volume confirmation to avoid false signals. Targets 20-40 trades/year.
Works in both bull and bear via trend-adaptive mean reversion.
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
    
    # Get 1-day data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot levels for current day
        # Need previous day's OHLC (1d data)
        day_idx = i // 4  # 4 = 24h / 6h (6h bars per day)
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
        r3 = pc + (range_val * 1.1 / 4)   # R3 = C + 1.1*(H-L)/4
        s3 = pc - (range_val * 1.1 / 4)   # S3 = C - 1.1*(H-L)/4
        
        # Trend direction from 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: >1.5x 20-period MA
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Fade logic: long at S3 in uptrend, short at R3 in downtrend
        long_setup = close[i] <= s3 and trend_up and vol_confirm
        short_setup = close[i] >= r3 and trend_down and vol_confirm
        
        # Exit logic: mean reversion to midpoint or trend breakdown
        midpoint = (ph + pl) / 2
        long_exit = close[i] >= midpoint or not trend_up
        short_exit = close[i] <= midpoint or not trend_down
        
        if long_setup and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_setup and position >= 0:
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

name = "6h_Camarilla_R3_S3_Fade_1dTrend"
timeframe = "6h"
leverage = 1.0