#!/usr/bin/env python3
"""
6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: On 6h timeframe, buying breakouts above weekly pivot R1 or selling breakdowns below weekly pivot S1, filtered by weekly trend (price above/below weekly EMA50) and volume confirmation, captures institutional breakout moves in both bull and bear markets. Weekly pivot provides institutional reference levels; EMA50 filter ensures trading with higher timeframe trend; volume confirms participation. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for pivot points and trend (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly pivot points calculation (standard formula)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Weekly EMA50 for trend filter
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False).mean().values
    
    # Align weekly data to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_weekly, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_weekly, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_weekly, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_weekly, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_weekly, s2_w)
    ema50_w_aligned = align_htf_to_ltf(prices, df_weekly, ema50_w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require 1.5x average volume
    
    # 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For weekly EMA50 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: breakdown below weekly S1 OR stoploss
            if (close[i] <= s1_w_aligned[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: breakout above weekly R1 OR stoploss
            if (close[i] >= r1_w_aligned[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot filter + volume
            # Long: break above Donchian high AND price above weekly R1 AND weekly uptrend
            long_breakout = high[i] > donchian_high[i]
            long_pivot_filter = close[i] > r1_w_aligned[i]
            long_trend = close[i] > ema50_w_aligned[i]
            
            # Short: break below Donchian low AND price below weekly S1 AND weekly downtrend
            short_breakout = low[i] < donchian_low[i]
            short_pivot_filter = close[i] < s1_w_aligned[i]
            short_trend = close[i] < ema50_w_aligned[i]
            
            if long_breakout and long_pivot_filter and long_trend and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and short_pivot_filter and short_trend and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals