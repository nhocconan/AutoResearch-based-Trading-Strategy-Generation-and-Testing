#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter_1wTrend_v1
Hypothesis: Ichimoku TK cross on 6h with 1d cloud filter (price above/below cloud) and 1w trend filter (EMA50) to capture strong trends while avoiding chop. Weekly EMA50 ensures we only trade in the direction of the higher timeframe trend, reducing false signals in sideways markets. Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud, 1w for trend)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 52 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 6h Ichimoku components (TK cross) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2.0
    
    # === 1d Ichimoku Cloud (Senkou Span A & B) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Senkou Span A = (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    highest_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    lowest_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (highest_tenkan_1d + lowest_tenkan_1d) / 2.0
    
    highest_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    lowest_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (highest_kijun_1d + lowest_kijun_1d) / 2.0
    
    senkou_span_a = (tenkan_1d + kijun_1d) / 2.0
    
    # Senkou Span B = (52-period high + 52-period low)/2 plotted 26 periods ahead
    highest_kijun_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    lowest_kijun_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (highest_kijun_52_1d + lowest_kijun_52_1d) / 2.0
    
    # Align cloud to 6h (Senkou Span A/B are plotted 26 periods ahead, so we need to shift back)
    # align_htf_to_ltf handles the completed-bar timing, so we align the values as calculated
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # The cloud is between Senkou Span A and B
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        
        # TK cross signals
        tk_cross_up = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1] if i > 0 else False
        tk_cross_down = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1] if i > 0 else False
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top_val
        price_below_cloud = price < cloud_bottom_val
        
        # Trend filter: price above/below weekly EMA50
        uptrend_filter = price > ema_50_1w_val
        downtrend_filter = price < ema_50_1w_val
        
        if position == 0:
            # Long: TK cross up, price above cloud, weekly uptrend
            long_condition = tk_cross_up and price_above_cloud and uptrend_filter
            # Short: TK cross down, price below cloud, weekly downtrend
            short_condition = tk_cross_down and price_below_cloud and downtrend_filter
            
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
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit conditions
            if position == 1:
                # Exit: TK cross down OR price falls below cloud bottom
                exit_condition = tk_cross_down or (price < cloud_bottom_val)
                if exit_condition:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit: TK cross up OR price rises above cloud top
                exit_condition = tk_cross_up or (price > cloud_top_val)
                if exit_condition:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloudFilter_1wTrend_v1"
timeframe = "6h"
leverage = 1.0