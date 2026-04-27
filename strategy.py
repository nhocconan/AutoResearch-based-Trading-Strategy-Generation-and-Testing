#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3_S3_Breakout_Trend_Filter
Breakout strategy using weekly pivot levels with trend filtering.
Long when price breaks above weekly R3 with bullish trend (price > weekly EMA50).
Short when price breaks below weekly S3 with bearish trend (price < weekly EMA50).
Exit when price returns to weekly pivot (PP) or trend reverses.
Uses volume confirmation to avoid false breakouts.
Target: 15-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivots(high, low, close):
    """Calculate weekly pivot points: R3, S3, PP"""
    H = high
    L = low
    C = close
    PP = (H + L + C) / 3.0
    R3 = H + 2 * (PP - L)
    S3 = L - 2 * (H - PP)
    return PP, R3, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points: R3, S3, PP
    PP_weekly, R3_weekly, S3_weekly = calculate_pivots(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to 6h timeframe
    PP_weekly_aligned = align_htf_to_ltf(prices, df_weekly, PP_weekly)
    R3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, R3_weekly)
    S3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, S3_weekly)
    
    # Weekly EMA50 for trend filter
    ema_weekly_period = 50
    ema_weekly = np.full(len(weekly_close), np.nan)
    if len(weekly_close) >= ema_weekly_period:
        ema_weekly[ema_weekly_period - 1] = np.mean(weekly_close[:ema_weekly_period])
        for i in range(ema_weekly_period, len(weekly_close)):
            ema_weekly[i] = (weekly_close[i] * (2 / (ema_weekly_period + 1)) + 
                             ema_weekly[i - 1] * (1 - (2 / (ema_weekly_period + 1))))
    
    # Align weekly EMA50 to 6h timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    if n >= vol_ma_period:
        vol_ma[vol_ma_period - 1] = np.mean(volume[:vol_ma_period])
        for i in range(vol_ma_period, n):
            vol_ma[i] = (volume[i] * (2 / (vol_ma_period + 1)) + 
                         vol_ma[i - 1] * (1 - (2 / (vol_ma_period + 1))))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly data, EMA, and volume MA
    start_idx = max(ema_weekly_period - 1, vol_ma_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(PP_weekly_aligned[i]) or np.isnan(R3_weekly_aligned[i]) or 
            np.isnan(S3_weekly_aligned[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        pp = PP_weekly_aligned[i]
        r3 = R3_weekly_aligned[i]
        s3 = S3_weekly_aligned[i]
        ema_weekly_val = ema_weekly_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_ok = vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above weekly R3 with bullish trend and volume
            if (price > r3 and price > ema_weekly_val and volume_ok):
                signals[i] = size
                position = 1
            # Short: price breaks below weekly S3 with bearish trend and volume
            elif (price < s3 and price < ema_weekly_val and volume_ok):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly PP or trend turns bearish
            if (price <= pp or price < ema_weekly_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to weekly PP or trend turns bullish
            if (price >= pp or price > ema_weekly_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_R3_S3_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0