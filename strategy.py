#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R3/S3) breakout with 1d trend filter and volume confirmation.
Enters long when price breaks above daily R3 with above-average volume and daily uptrend.
Enters short when price breaks below daily S3 with above-average volume and daily downtrend.
Uses daily timeframe for structure and trend, 4h for execution to reduce noise and capture swing moves.
Designed to work in both bull and bear markets by following the daily trend and requiring volume confirmation.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot points (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Support and resistance levels (Camarilla)
    S1 = close_1d - (high_1d - low_1d) * 1.0 / 12
    S2 = close_1d - (high_1d - low_1d) * 2.0 / 12
    S3 = close_1d - (high_1d - low_1d) * 3.0 / 12
    S4 = close_1d - (high_1d - low_1d) * 4.0 / 12
    R1 = close_1d + (high_1d - low_1d) * 1.0 / 12
    R2 = close_1d + (high_1d - low_1d) * 2.0 / 12
    R3 = close_1d + (high_1d - low_1d) * 3.0 / 12
    R4 = close_1d + (high_1d - low_1d) * 4.0 / 12
    
    # Align daily pivot levels to 4h timeframe
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    
    # Get 4h data for volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h volume MA(20)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily pivot levels, volume MA, and daily EMA
    start_idx = max(20, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(S3_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        trend_1d = ema_34_1d_aligned[i]
        
        # Current daily pivot levels
        S3_now = S3_aligned[i]
        R3_now = R3_aligned[i]
        
        # Volume filter: volume > 1.5x 4h average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Daily Camarilla R3/S3 breakout with volume and daily trend alignment
        if position == 0:
            # Long: price breaks above R3 with volume + daily uptrend
            if price_now > R3_now and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with volume + daily downtrend
            elif price_now < S3_now and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to daily pivot or daily trend turns down
            pivot_1d = (high_1d + low_1d + close_1d) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            pivot_now = pivot_aligned[i]
            if price_now <= pivot_now or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to daily pivot or daily trend turns up
            pivot_1d = (high_1d + low_1d + close_1d) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            pivot_now = pivot_aligned[i]
            if price_now >= pivot_now or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0