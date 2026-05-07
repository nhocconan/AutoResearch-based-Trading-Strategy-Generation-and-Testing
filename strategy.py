#!/usr/bin/env python3
"""
12h_Pivot_Breakout_1dTrend_Volume
Hypothesis: Price breaking above/below daily Camarilla pivot levels (S3/R3) with 1d trend filter and volume spike (>1.5x 24-period average). 
Pivot levels act as key support/resistance; breaks indicate institutional interest. 1d EMA50 ensures alignment with higher timeframe trend.
Volume spike confirms breakout strength. Designed for low trade frequency (12-25/year) with clear trend-following logic.
Works in bull/bear markets by requiring trend alignment and volatility-based confirmation.
"""

name = "12h_Pivot_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get 1d data for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (S3, R3) from previous day
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    hl_range = daily_high - daily_low
    r3 = daily_close + 1.1 * hl_range
    s3 = daily_close - 1.1 * hl_range
    
    # Align pivot levels to 12h timeframe (previous day's levels available at open)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 24-period average (2 * 12h periods)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.divide(volume, vol_ma24, out=np.zeros_like(volume), where=vol_ma24!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = daily_close[i] > ema_50_1d[i]  # Use daily close for trend
        trend_down = daily_close[i] < ema_50_1d[i]
        
        if position == 0:
            # Long: Close breaks above R3 with uptrend and volume spike
            if close[i] > r3_aligned[i] and trend_up and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 with downtrend and volume spike
            elif close[i] < s3_aligned[i] and trend_down and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close crosses below S3 or trend turns down
            if close[i] < s3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close crosses above R3 or trend turns up
            if close[i] > r3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals