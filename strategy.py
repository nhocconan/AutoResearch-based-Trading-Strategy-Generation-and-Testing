#!/usr/bin/env python3
"""
6h Weekly Pivot + Donchian Breakout with Volume Confirmation
Hypothesis: Weekly pivot levels act as strong support/resistance. Breakouts above weekly R4 or below S4 with volume confirmation and 1d trend alignment capture significant moves. Works in bull (breakouts above R4 in uptrend) and bear (breakdowns below S4 in downtrend). Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_breakout_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    # Load weekly and daily data (once before loop)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly OHLC from weekly data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot points: P = (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Calculate support/resistance levels
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    weekly_range = weekly_high - weekly_low
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + weekly_range
    s2 = weekly_pivot - weekly_range
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    r4 = r3 + weekly_range
    s4 = s3 - weekly_range
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20 periods)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(200, lookback)  # Ensure sufficient data
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below weekly pivot OR stoploss
            if (close[i] <= pivot_aligned[i] or
                close[i] <= entry_price - 2.5 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above weekly pivot OR stoploss
            if (close[i] >= pivot_aligned[i] or
                close[i] >= entry_price + 2.5 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout beyond weekly R4/S4 with volume and trend alignment
            # Uptrend: price > daily EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            # Downtrend: price < daily EMA50
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: break above weekly R4 with volume
            long_breakout = (high[i] > r4_aligned[i]) and vol_filter[i] and uptrend
            # Short: break below weekly S4 with volume
            short_breakout = (low[i] < s4_aligned[i]) and vol_filter[i] and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals