#!/usr/bin/env python3
"""
6h_WeeklyPivot_Position_With_1dTrend
Hypothesis: Trade long/short based on weekly pivot position (above/below weekly pivot) confirmed by daily trend (price vs daily EMA50) and volume spike. 
In bull markets (price > daily EMA50), go long when price is above weekly pivot with volume confirmation. 
In bear markets (price < daily EMA50), go short when price is below weekly pivot with volume confirmation.
Weekly pivot provides key institutional levels; daily trend filters for market regime; volume confirms institutional participation.
Designed for 15-25 trades/year to minimize fee drag. Works in bull/bear via daily trend filter.
"""

name = "6h_WeeklyPivot_Position_With_1dTrend"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot for each week
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Align weekly pivot to 6h timeframe (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot, additional_delay_bars=0)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 6h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(close_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Price relative to weekly pivot
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: above weekly pivot in uptrend with volume spike
            if price_above_pivot and trend_up and vol_ratio[i] > 1.8:
                signals[i] = 0.25
                position = 1
            # Short: below weekly pivot in downtrend with volume spike
            elif price_below_pivot and trend_down and vol_ratio[i] > 1.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price drops below weekly pivot or trend turns down
            if not price_above_pivot or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above weekly pivot or trend turns up
            if not price_below_pivot or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals