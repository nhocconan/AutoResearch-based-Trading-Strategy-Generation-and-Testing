#!/usr/bin/env python3
# 6h_WeeklyTrend_DailyPullback
# Hypothesis: On 6h chart, go long when price pulls back to 1d VWAP during a 1w uptrend (price > weekly VWAP), short when price rallies to 1d VWAP during a 1w downtrend. Uses volume confirmation to avoid false signals.
# Designed for 6h timeframe with 1w trend filter and 1d VWAP mean reversion. Works in both bull and bear markets by trading with the weekly trend.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WeeklyTrend_DailyPullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w and 1d data for trend and VWAP
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1w VWAP for trend filter ---
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    vwap_num_1w = np.cumsum(typical_price_1w * df_1w['volume'].values)
    vwap_den_1w = np.cumsum(df_1w['volume'].values)
    vwap_1w = vwap_num_1w / vwap_den_1w
    vwap_1w[vwap_den_1w == 0] = np.nan
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # --- 1d VWAP for entry signal ---
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_num_1d = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_den_1d = np.cumsum(df_1d['volume'].values)
    vwap_1d = vwap_num_1d / vwap_den_1d
    vwap_1d[vwap_den_1d == 0] = np.nan
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # --- Volume Filter: above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after enough data for volume median
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            if position != 0:
                # Exit on opposite VWAP touch
                if position == 1 and close_6h[i] <= vwap_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= vwap_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1w trend
        trend_up = close_6h[i] > vwap_1w_aligned[i]
        trend_down = close_6h[i] < vwap_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries: pullback to 1d VWAP in direction of 1w trend
            if trend_up and vol_ok and close_6h[i] <= vwap_1d_aligned[i] * 1.005:  # Allow small buffer
                # Long: price at or below 1d VWAP during 1w uptrend
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            elif trend_down and vol_ok and close_6h[i] >= vwap_1d_aligned[i] * 0.995:  # Allow small buffer
                # Short: price at or above 1d VWAP during 1w downtrend
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Exit when price crosses 1d VWAP in opposite direction
            if position == 1:
                if close_6h[i] <= vwap_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close_6h[i] >= vwap_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals