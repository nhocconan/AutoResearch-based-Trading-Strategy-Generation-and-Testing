#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filtered_Trend
Hypothesis: Use Ichimoku Cloud from 1d timeframe to determine trend direction and filter entries on 6h. Enter long when price is above cloud and Tenkan-Kijun cross is bullish; enter short when price is below cloud and cross is bearish. Use volume confirmation (volume > 1.5x 20-period average) to avoid false signals. Designed for 15-30 trades/year per symbol to avoid fee drag while capturing major trends. Works in both bull and bear markets by following the 1d Ichimoku trend.
"""

name = "6h_Ichimoku_Cloud_Filtered_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Ichimoku Cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Ichimoku Cloud Components ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Align Ichimoku components to 6h
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # --- 6h Volume Average for confirmation ---
    vol_avg_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period for Ichimoku (52 periods)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or
            np.isnan(vol_avg_6h[i])):
            if position != 0:
                # Simple stop: exit if price crosses opposite cloud boundary
                if position == 1 and close_6h[i] < senkou_span_b_6h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] > senkou_span_a_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        lower_cloud = np.minimum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        # Determine Tenkan-Kijun cross
        tk_cross = tenkan_sen_6h[i] - kijun_sen_6h[i]
        tk_cross_prev = tenkan_sen_6h[i-1] - kijun_sen_6h[i-1] if i > 0 else 0
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_6h[i] > 1.5 * vol_avg_6h[i]
        
        if position == 0:
            # Look for entries: price outside cloud + TK cross in same direction
            if vol_confirm:
                # Bullish: price above cloud + TK cross bullish (Tenkan > Kijun)
                if close_6h[i] > upper_cloud and tk_cross > 0 and tk_cross_prev <= 0:
                    signals[i] = 0.25  # long
                    position = 1
                    entry_price = close_6h[i]
                # Bearish: price below cloud + TK cross bearish (Tenkan < Kijun)
                elif close_6h[i] < lower_cloud and tk_cross < 0 and tk_cross_prev >= 0:
                    signals[i] = -0.25  # short
                    position = -1
                    entry_price = close_6h[i]
        else:
            # Manage existing position: exit when price re-enters cloud
            if position == 1:
                # Long: exit if price falls below cloud
                if close_6h[i] < lower_cloud:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit if price rises above cloud
                if close_6h[i] > upper_cloud:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals