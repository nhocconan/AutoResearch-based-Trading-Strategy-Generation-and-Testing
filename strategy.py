#!/usr/bin/env python3
"""
6h_HTF_1d_Ichimoku_Cloud_Trend_V1
Hypothesis: 6h price breaking above/below Ichimoku cloud (from 1d) with TK cross confirmation and volume filter. 
Ichimoku cloud acts as dynamic support/resistance; TK cross confirms momentum. 
Volume > 1.5x 20-period MA reduces false breakouts. Works in bull/bear by only taking trades aligned with 1d cloud color (green=long bias, red=short bias).
Target: 12-37 trades/year (50-150 total over 4 years).
Uses 6h primary timeframe with 1d HTF for Ichimoku cloud and TK cross.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B (26*2)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = (pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                     pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Current cloud (Senkou Span A/B shifted back 26 periods to align with current price)
    # We need the cloud values from 26 periods ago for current price
    if len(senkou_span_a) < 26 or len(senkou_span_b) < 26:
        return np.zeros(n)
    senkou_span_a_lagged = np.roll(senkou_span_a, 26)
    senkou_span_b_lagged = np.roll(senkou_span_b, 26)
    # First 26 values are invalid due to roll
    senkou_span_a_lagged[:26] = np.nan
    senkou_span_b_lagged[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_span_a_lagged, senkou_span_b_lagged)
    cloud_bottom = np.minimum(senkou_span_a_lagged, senkou_span_b_lagged)
    cloud_green = senkou_span_a_lagged > senkou_span_b_lagged  # True = bullish cloud
    
    # TK Cross (Tenkan-sen crossing Kijun-sen)
    tk_cross = tenkan_sen - kijun_sen
    tk_cross_above = tk_cross > 0  # Tenkan above Kijun = bullish
    tk_cross_below = tk_cross < 0  # Tenkan below Kijun = bearish
    
    # Align all 1d indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    cloud_green_aligned = align_htf_to_ltf(prices, df_1d, cloud_green.astype(float))  # bool to float
    tk_cross_above_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_above.astype(float))
    tk_cross_below_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_below.astype(float))
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if indicators not ready
        if (np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or 
            np.isnan(tk_cross_above_aligned[i]) or np.isnan(tk_cross_below_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        price_above_cloud = price > cloud_top_aligned[i]
        price_below_cloud = price < cloud_bottom_aligned[i]
        in_cloud = (price >= cloud_bottom_aligned[i]) & (price <= cloud_top_aligned[i])
        
        if position == 0:
            # Long: price breaks above cloud + TK cross bullish + volume + green cloud bias
            if price_above_cloud and tk_cross_above_aligned[i] > 0.5 and vol_ok and cloud_green_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud + TK cross bearish + volume + red cloud bias
            elif price_below_cloud and tk_cross_below_aligned[i] > 0.5 and vol_ok and cloud_green_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below cloud bottom OR TK cross turns bearish
            if price < cloud_bottom_aligned[i] or tk_cross_above_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above cloud top OR TK cross turns bullish
            if price > cloud_top_aligned[i] or tk_cross_below_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_1d_Ichimoku_Cloud_Trend_V1"
timeframe = "6h"
leverage = 1.0