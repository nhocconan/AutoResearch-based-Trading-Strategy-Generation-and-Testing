#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_1wTrend_v1
Hypothesis: 6h Ichimoku cloud breakouts with 1w trend filter (price above/below weekly Kumo) for regime alignment. Uses TK cross for entry timing, cloud thickness for volatility filter, and discrete sizing (0.25). Target: 50-150 total trades over 4 years for BTC/ETH/SOL by combining multiple confirmation layers. Works in bull/bear via weekly trend filter that adapts to longer-term structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span"""
    if len(high) < kijun:
        return (np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float))
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max().values
    period9_low = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=kijun, min_periods=kijun).max().values
    period26_low = pd.Series(low).rolling(window=kijun, min_periods=kijun).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=senkou, min_periods=senkou).max().values
    period52_low = pd.Series(low).rolling(window=senkou, min_periods=senkou).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou_span = np.roll(close, -kijun)  # shifted left (future values)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Ichimoku for trend regime ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(
        high_1w, low_1w, close_1w
    )
    
    # Kumo (cloud) top and bottom
    kumO_top_1w = np.maximum(senkou_a_1w, senkou_b_1w)
    kumO_bottom_1w = np.minimum(senkou_a_1w, senkou_b_1w)
    kumO_top_1w_aligned = align_htf_to_ltf(prices, df_1w, kumO_top_1w)
    kumO_bottom_1w_aligned = align_htf_to_ltf(prices, df_1w, kumO_bottom_1w)
    
    # Cloud thickness (volatility filter)
    cloud_thickness_1w = kumO_top_1w - kumO_bottom_1w
    cloud_thickness_aligned = align_htf_to_ltf(prices, df_1w, cloud_thickness_1w)
    
    # === 6h Ichimoku for entry signals ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h, chikou_6h = calculate_ichimoku(
        high, low, close
    )
    
    # Kumo (cloud) top and bottom for 6h
    kumO_top_6h = np.maximum(senkou_a_6h, senkou_b_6h)
    kumO_bottom_6h = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # TK Cross (Tenkan-sen crosses Kijun-sen)
    tk_cross_up = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(52, n):  # wait for Ichimoku to warm up
        # Skip if indicators not ready
        if (np.isnan(kumO_top_1w_aligned[i]) or np.isnan(kumO_bottom_1w_aligned[i]) or
            np.isnan(cloud_thickness_aligned[i]) or np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or np.isnan(kumO_top_6h[i]) or np.isnan(kumO_bottom_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        kumO_top_1w = kumO_top_1w_aligned[i]
        kumO_bottom_1w = kumO_bottom_1w_aligned[i]
        cloud_thickness = cloud_thickness_aligned[i]
        tenkan = tenkan_6h[i]
        kijun = kijun_6h[i]
        kumO_top = kumO_top_6h[i]
        kumO_bottom = kumO_bottom_6h[i]
        tk_up = tk_cross_up[i]
        tk_down = tk_cross_down[i]
        
        # Trend regime: price relative to weekly Kumo
        is_bull = price > kumO_top_1w
        is_bear = price < kumO_bottom_1w
        
        # Cloud thickness filter: avoid choppy clouds
        thick_cloud = cloud_thickness > (0.01 * price)  # cloud > 1% of price
        
        if position == 0:
            if is_bull and thick_cloud:
                # Bull regime: long on TK cross up, price above cloud
                long_condition = tk_up and (price > kumO_top)
            elif is_bear and thick_cloud:
                # Bear regime: short on TK cross down, price below cloud
                short_condition = tk_down and (price < kumO_bottom)
            else:
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit conditions
            if position == 1:
                # Exit long: TK cross down OR price breaks below cloud bottom
                if tk_down or (price < kumO_bottom):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: TK cross up OR price breaks above cloud top
                if tk_up or (price > kumO_top):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1wTrend_v1"
timeframe = "6h"
leverage = 1.0