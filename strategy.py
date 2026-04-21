#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Regime_v1
Hypothesis: Ichimoku cloud breakout (Tenkan/Kijun cross) with weekly trend filter (price above/below weekly cloud) and 6h volume confirmation reduces false signals while capturing strong trends. Designed for low trade frequency (~50-100/year) to minimize fee drag and improve generalization across bull/bear markets. Works in both regimes: in uptrends, longs from bullish TK cross above weekly cloud; in downtrends, shorts from bearish TK cross below weekly cloud.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 2 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === Ichimoku components on 6h (primary timeframe) ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    hh_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    ll_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (hh_tenkan + ll_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    hh_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    ll_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (hh_kijun + ll_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    hh_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    ll_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (hh_senkou_b + ll_senkou_b) / 2
    
    # Chikou Span (Lagging Span): not used for signals (look-ahead)
    
    # === Weekly trend filter: price relative to weekly cloud ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Tenkan-sen and Kijun-sen
    hh_tenkan_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    ll_tenkan_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (hh_tenkan_1w + ll_tenkan_1w) / 2
    
    hh_kijun_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    ll_kijun_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (hh_kijun_1w + ll_kijun_1w) / 2
    
    # Weekly Senkou Span A and B
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    hh_senkou_b_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    ll_senkou_b_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (hh_senkou_b_1w + ll_senkou_b_1w) / 2
    
    # Weekly cloud top and bottom
    weekly_cloud_top = np.maximum(senkou_a_1w, senkou_b_1w)
    weekly_cloud_bottom = np.minimum(senkou_a_1w, senkou_b_1w)
    
    # Align weekly HTF indicators to 6h
    weekly_cloud_top_aligned = align_htf_to_ltf(prices, df_1w, weekly_cloud_top)
    weekly_cloud_bottom_aligned = align_htf_to_ltf(prices, df_1w, weekly_cloud_bottom)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    
    # === Daily volume confirmation (to avoid low-volume breakouts) ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(weekly_cloud_top_aligned[i]) or np.isnan(weekly_cloud_bottom_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        tk_cross = tenkan[i] - kijun[i]
        tk_cross_prev = tenkan[i-1] - kijun[i-1] if i > 0 else 0
        vol_confirm = vol_ratio_1d_aligned[i]
        weekly_top = weekly_cloud_top_aligned[i]
        weekly_bottom = weekly_cloud_bottom_aligned[i]
        
        if position == 0:
            # Long: bullish TK cross (Tenkan crosses above Kijun) + price above weekly cloud + volume confirmation
            if tk_cross > 0 and tk_cross_prev <= 0 and price_close > weekly_top and vol_confirm > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: bearish TK cross (Tenkan crosses below Kijun) + price below weekly cloud + volume confirmation
            elif tk_cross < 0 and tk_cross_prev >= 0 and price_close < weekly_bottom and vol_confirm > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit on opposite TK cross or price re-enters weekly cloud
            if position == 1:
                # Exit long: bearish TK cross OR price drops below weekly cloud bottom
                if tk_cross < 0 and tk_cross_prev >= 0:
                    signals[i] = 0.0
                    position = 0
                elif price_close < weekly_bottom:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: bullish TK cross OR price rises above weekly cloud top
                if tk_cross > 0 and tk_cross_prev <= 0:
                    signals[i] = 0.0
                    position = 0
                elif price_close > weekly_top:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Regime_v1"
timeframe = "6h"
leverage = 1.0