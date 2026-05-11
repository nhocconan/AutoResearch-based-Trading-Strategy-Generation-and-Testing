#!/usr/bin/env python3
# 6h_Weekly_Pivot_Trend_Filter
# Hypothesis: Uses weekly pivot points (from weekly OHLC) as dynamic support/resistance
# levels. In uptrends (price > weekly pivot), go long on retracement to pivot with
# volume confirmation. In downtrends (price < weekly pivot), go short on bounce
# off pivot with volume confirmation. Weekly pivots provide structure that works
# in both bull and bear markets by adapting to the current weekly range.
# Timeframe: 6h, uses 1w data for pivots, 1d for trend filter.

name = "6h_Weekly_Pivot_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation and daily for trend
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Pivot Calculation (using prior week's OHLC) ---
    # Weekly high, low, close from completed weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point: (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align pivot to 6s timeframe (using completed weekly bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # --- Daily EMA50 for trend filter ---
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Volume confirmation: current volume > 1.5x 20-period average ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price relative to daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: uptrend + price at or below pivot + volume confirmation
            if uptrend and close[i] <= pivot_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price at or above pivot + volume confirmation
            elif downtrend and close[i] >= pivot_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses above pivot OR trend changes
                if close[i] > pivot_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses below pivot OR trend changes
                if close[i] < pivot_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals