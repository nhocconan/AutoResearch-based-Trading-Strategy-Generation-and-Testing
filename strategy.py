#!/usr/bin/env python3
"""
4h_1d_Ichimoku_Cloud_Trend_Follower
Hypothesis: Uses daily Ichimoku Cloud to determine primary trend direction (price above/below cloud),
with 4h Tenkan-Kijun cross as entry signal and volume confirmation. Designed to work in both bull and bear markets by following higher-timeframe trend while using lower timeframe for precise entries. Targets low trade frequency (19-50/year) via daily trend filter.
"""

name = "4h_1d_Ichimoku_Cloud_Trend_Follower"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close).shift(26)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Ichimoku for Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align daily Ichimoku to 4h timeframe
    tenkan_1d_4h = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_4h = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_4h = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_4h = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_4h = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # --- 4h Tenkan-Kijun Cross for Entry Timing ---
    tenkan_4h, kijun_4h, _, _, _ = calculate_ichimoku(high, low, close)
    
    # Calculate Tenkan-Kijun cross signals
    tk_cross_above = (tenkan_4h > kijun_4h) & (tenkan_4h.shift(1) <= kijun_4h.shift(1))
    tk_cross_below = (tenkan_4h < kijun_4h) & (tenkan_4h.shift(1) >= kijun_4h.shift(1))
    
    # --- Volume Spike Detection (20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_4h[i]) or np.isnan(kijun_1d_4h[i]) or 
            np.isnan(senkou_a_1d_4h[i]) or np.isnan(senkou_b_1d_4h[i]) or
            np.isnan(tenkan_4h[i]) or np.isnan(kijun_4h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        green_cloud = senkou_a_1d_4h[i] > senkou_b_1d_4h[i]
        red_cloud = senkou_a_1d_4h[i] < senkou_b_1d_4h[i]
        
        above_cloud = close[i] > max(senkou_a_1d_4h[i], senkou_b_1d_4h[i])
        below_cloud = close[i] < min(senkou_a_1d_4h[i], senkou_b_1d_4h[i])
        in_cloud = not above_cloud and not below_cloud
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price above green cloud + TK cross up + volume
            if (above_cloud and green_cloud and 
                tk_cross_above[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price below red cloud + TK cross down + volume
            elif (below_cloud and red_cloud and 
                  tk_cross_below[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite cloud break or TK cross in opposite direction
            if position == 1:
                # Exit long: price breaks below cloud OR TK cross down
                if below_cloud or tk_cross_below[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above cloud OR TK cross up
                if above_cloud or tk_cross_above[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals