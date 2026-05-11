#!/usr/bin/env python3
"""
6h_1d_Weekly_Trend_Pullback
Hypothesis: Price pulls back to EMA200 on 6h during a strong weekly trend (above/below weekly EMA50), entering in direction of weekly trend. Uses volume confirmation to avoid false breakouts. Designed to work in both bull and bear markets by following weekly trend. Targets ~15-25 trades/year.
"""

name = "6h_1d_Weekly_Trend_Pullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for EMA200 (trend filter on 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d EMA200 for 6h trend filter ---
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # --- 1w EMA50 for weekly trend direction ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- Volume filter: above 20-period median ---
    vol_median = pd.Series(volume_6h).rolling(window=20, min_periods=10).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period (200 for EMA200)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                # Check if price has moved against position significantly
                if position == 1 and close_6h[i] < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine weekly trend
        weekly_uptrend = close_6h[i] > ema50_1w_aligned[i]
        weekly_downtrend = close_6h[i] < ema50_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume_6h[i] > vol_median[i]
        
        if position == 0:
            # Look for pullback to EMA200 in direction of weekly trend
            if weekly_uptrend and close_6h[i] <= ema200_1d_aligned[i] * 1.02 and \
               close_6h[i] >= ema200_1d_aligned[i] * 0.98 and vol_ok:
                # Long: pullback to EMA200 during weekly uptrend with volume
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            elif weekly_downtrend and close_6h[i] >= ema200_1d_aligned[i] * 0.98 and \
                 close_6h[i] <= ema200_1d_aligned[i] * 1.02 and vol_ok:
                # Short: pullback to EMA200 during weekly downtrend with volume
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Exit when price crosses EMA200 against position
            if position == 1:
                if close_6h[i] < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close_6h[i] > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals