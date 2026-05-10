#!/usr/bin/env python3
# 1D_Premium_Discount_Zone_1wTrend_Volume
# Hypothesis: Price respecting weekly premium/discount zones (based on weekly range) 
# combined with weekly trend filter (21-period EMA) and daily volume confirmation 
# provides high-probability entries. In uptrends, buy at discount zone; in downtrends, 
# sell at premium zone. Works in both bull and bear markets by aligning with weekly 
# structure and avoiding chop via volume filter. Target: 15-25 trades/year.

name = "1D_Premium_Discount_Zone_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and range
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for volume and reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Weekly high/low for premium/discount zones
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_range = weekly_high - weekly_low
    # Discount zone: 0-50% of weekly range from low
    # Premium zone: 50-100% of weekly range from low (or 0-50% from high)
    discount_zone = weekly_low + weekly_range * 0.5
    premium_zone = weekly_low + weekly_range * 0.5  # Same line, but we'll use context
    
    # Align weekly levels to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_range_aligned = align_htf_to_ltf(prices, df_1w, weekly_range)
    discount_zone_aligned = align_htf_to_ltf(prices, df_1w, discount_zone)
    premium_zone_aligned = align_htf_to_ltf(prices, df_1w, premium_zone)
    
    # Daily volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA21 (21), weekly range (implicitly), volume MA (20)
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price at or below discount zone (50% level) + volume
            if uptrend and close[i] <= discount_zone_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price at or above premium zone (50% level) + volume
            elif downtrend and close[i] >= premium_zone_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price moves above premium zone (take profit)
            if not uptrend or close[i] >= premium_zone_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price moves below discount zone (take profit)
            if not downtrend or close[i] <= discount_zone_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals