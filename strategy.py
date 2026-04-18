#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_With_Daily_Trend
Hypothesis: Weekly pivot levels (R4/S4) act as strong support/resistance. Breakouts above R4 or below S4 with daily EMA trend continuation capture institutional momentum. Works in bull/bear markets by following major breakouts. Target: 10-25 trades/year (40-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (use daily as proxy for weekly pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data (approximation)
    # Use weekly high/low/close from resampled daily data
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().values  # Approx weekly
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().values
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Calculate R4 and S4 levels (more aggressive breakout levels)
    weekly_range = weekly_high - weekly_low
    r4 = weekly_pivot + 3 * weekly_range
    s4 = weekly_pivot - 3 * weekly_range
    
    # Align weekly levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Daily EMA trend filter (34-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for weekly calculations
    
    for i in range(start_idx, n):
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema_1d_6h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_6h[i]
        r4_level = r4_6h[i]
        s4_level = s4_6h[i]
        
        if position == 0:
            # Long: break above R4 with volume and daily uptrend
            if price > r4_level and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S4 with volume and daily downtrend
            elif price < s4_level and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price falls below weekly pivot or trend reverses
            if price < weekly_pivot_6h[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price rises above weekly pivot or trend reverses
            if price > weekly_pivot_6h[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Breakout_With_Daily_Trend"
timeframe = "6h"
leverage = 1.0