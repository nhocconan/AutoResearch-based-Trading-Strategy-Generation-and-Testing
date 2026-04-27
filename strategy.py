#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_ADXFilter
Hypothesis: 6h strategy using Ichimoku cloud from 1d for trend direction, 1w ADX for regime filter, and Tenkan-Kijun cross for precise entries. 
Enter long when price is above cloud (bullish), Tenkan crosses above Kijun, and 1w ADX > 25 (strong trend). 
Enter short when price is below cloud (bearish), Tenkan crosses below Kijun, and 1w ADX > 25. 
Exit when price re-enters cloud or Tenkan-Kijun cross reverses. 
Designed for low-moderate trade frequency (~15-30/year) with discrete position sizing (0.25) to minimize fee drag while capturing strong trends.
Uses weekly ADX to avoid whipsaws in ranging markets, improving performance in both bull and bear regimes by only trading when 1w trend is strong.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Ichimoku components (1d)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    tenkan = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).mean() + 
              pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).mean()) / 2
    tenkan = tenkan.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).mean() + 
             pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).mean()) / 2
    kijun = kijun.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    senkou_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).mean() + 
                 pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).mean()) / 2)
    
    # Current cloud boundaries (Senkou Span A/B from 26 periods ago)
    # Since Senkou spans are plotted 26 periods ahead, current cloud is from 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Align Ichimoku components to 6h timeframe (completed 1d bars only)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_lagged)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_lagged)
    
    # 1w ADX for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+,
    period_adx = 14
    tr_smooth = pd.Series(tr).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    
    # Align 1w ADX to 6h timeframe (completed 1w bars only)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Ichimoku (52) + ADX (14+14=28) -> max 52
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        adx_val = adx_aligned[i]
        
        # Cloud boundaries: top is max(senkou_a, senkou_b), bottom is min(senkou_a, senkou_b)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Tenkan-Kijun cross detection (using previous bar values)
        if i > start_idx:
            prev_tenkan = tenkan_aligned[i-1]
            prev_kijun = kijun_aligned[i-1]
            tenkan_above_prev = prev_tenkan > prev_kijun
            tenkan_above_curr = tenkan_val > kijun_val
            tk_cross_up = (not tenkan_above_prev) and tenkan_above_cross
            tk_cross_down = tenkan_above_prev and (not tenkan_above_curr)
        else:
            tk_cross_up = False
            tk_cross_down = False
        
        if position == 0:
            # Look for entry: price outside cloud + TK cross in direction of price + strong 1w trend (ADX > 25)
            # Long: price above cloud, TK cross up, ADX > 25
            long_condition = (close_val > cloud_top) and tk_cross_up and (adx_val > 25)
            # Short: price below cloud, TK cross down, ADX > 25
            short_condition = (close_val < cloud_bottom) and tk_cross_down and (adx_val > 25)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price re-enters cloud OR TK cross down
            if (close_val <= cloud_top) or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters cloud OR TK cross up
            if (close_val >= cloud_bottom) or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_ADXFilter"
timeframe = "6h"
leverage = 1.0