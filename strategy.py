#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_WeeklyTrend
Hypothesis: Use weekly trend direction (price vs weekly EMA34) as bias, then trade Camarilla R3/S3 breakout on 1d timeframe with volume confirmation. Weekly trend provides multi-month context that works in both bull and bear markets, while Camarilla R3/S3 levels offer significant breakout points with favorable risk-reward. Target: 10-25 trades/year.
"""

name = "1d_Camarilla_R3_S3_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for trend bias
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = ema_34_1w > np.roll(ema_34_1w, 1)  # upward slope
    weekly_trend_down = ema_34_1w < np.roll(ema_34_1w, 1)  # downward slope
    
    # Align weekly trend to daily timeframe
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    range_prev = high_prev - low_prev
    
    # Camarilla R3 and S3 levels
    r3 = close_prev + 1.1 * range_prev * 1.1666  # (close + 1.1 * range * 1.1666)
    s3 = close_prev - 1.1 * range_prev * 1.1666  # (close - 1.1 * range * 1.1666)
    
    # Align Camarilla levels to daily timeframe (no additional delay needed)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (34 periods) and daily data
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend AND price breaks above R3 with volume
            if weekly_trend_up_aligned[i] and high[i] > r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend AND price breaks below S3 with volume
            elif weekly_trend_down_aligned[i] and low[i] < s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversal signal) or weekly trend turns down
            if low[i] < s3_aligned[i] or not weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 (reversal signal) or weekly trend turns up
            if high[i] > r3_aligned[i] or not weekly_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals