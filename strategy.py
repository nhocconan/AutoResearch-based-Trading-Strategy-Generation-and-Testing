# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h Camarilla Pivot Breakout with 1d Volume Confirmation and 1w Trend Filter
Enters long when price breaks above daily R1 with volume above average and weekly uptrend.
Enters short when price breaks below daily S1 with volume above average and weekly downtrend.
Uses 12h as primary timeframe with daily pivots for structure and weekly trend filter.
Designed to work in both bull and bear markets by following weekly trend and requiring volume confirmation.
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Resistance levels
    R1 = pivot + (range_1d * 1.0 / 12)
    R2 = pivot + (range_1d * 2.0 / 12)
    R3 = pivot + (range_1d * 3.0 / 12)
    R4 = pivot + (range_1d * 4.0 / 12)
    
    # Support levels
    S1 = pivot - (range_1d * 1.0 / 12)
    S2 = pivot - (range_1d * 2.0 / 12)
    S3 = pivot - (range_1d * 3.0 / 12)
    S4 = pivot - (range_1d * 4.0 / 12)
    
    # Align daily pivot levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 12h data for volume confirmation
    # Volume MA(34) on 12h timeframe
    vol_ma_34 = pd.Series(volume).rolling(window=34, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily pivots, weekly EMA, and volume MA
    start_idx = max(2, 34, 34)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_34[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_34[i]
        trend_1w = ema_34_1w_aligned[i]
        
        # Current daily pivot levels
        R1_now = R1_aligned[i]
        S1_now = S1_aligned[i]
        
        # Volume filter: volume > 1.2x 12h average
        vol_filter = vol_now > 1.2 * vol_ma
        
        # Entry conditions: Daily Camarilla breakout with volume and weekly trend alignment
        if position == 0:
            # Long: price breaks above R1 with volume + weekly uptrend
            if price_now > R1_now and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with volume + weekly downtrend
            elif price_now < S1_now and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or weekly trend turns down
            pivot_aligned = align_htf_to_ltf(prices, df_1d, (high_1d + low_1d + close_1d) / 3.0)
            pivot_now = pivot_aligned[i]
            if price_now <= pivot_now or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or weekly trend turns up
            pivot_aligned = align_htf_to_ltf(prices, df_1d, (high_1d + low_1d + close_1d) / 3.0)
            pivot_now = pivot_aligned[i]
            if price_now >= pivot_now or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dVolume_1wTrend"
timeframe = "12h"
leverage = 1.0