#!/usr/bin/env python3
"""
6h_1w_1d_Ichimoku_Cloud_Trend_v1
Hypothesis: Use Ichimoku Cloud on daily timeframe for trend direction and weekly pivot levels for entry/exit. 
Ichimoku provides multi-dimensional trend/ momentum signals that work in both bull and bear regimes.
Weekly pivots act as dynamic support/resistance for breakout entries.
Target: 15-30 trades/year (60-120 total over 4 years) by requiring strong alignment of trend and momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY ICHIMOKU COMPONENTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    if len(high_1d) >= period_tenkan:
        tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                      pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
        tenkan_sen = tenkan_sen.values
    else:
        tenkan_sen = np.full_like(high_1d, np.nan)
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    if len(high_1d) >= period_kijun:
        kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                     pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
        kijun_sen = kijun_sen.values
    else:
        kijun_sen = np.full_like(high_1d, np.nan)
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    if not np.all(np.isnan(tenkan_sen)) and not np.all(np.isnan(kijun_sen)):
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    else:
        senkou_span_a = np.full_like(high_1d, np.nan)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    if len(high_1d) >= period_senkou_b:
        senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                          pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2)
        senkou_span_b = senkou_span_b.values
    else:
        senkou_span_b = np.full_like(high_1d, np.nan)
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    if len(close_1d) >= 26:
        chikou_span = np.concatenate([np.full(26, np.nan), close_1d[:-26]])
    else:
        chikou_span = np.full_like(close_1d, np.nan)
    
    # === WEEKLY PIVOT LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Point
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly Support/Resistance Levels
    r1_1w = pivot_1w + range_1w
    s1_1w = pivot_1w - range_1w
    r2_1w = pivot_1w + (range_1w * 2)
    s2_1w = pivot_1w - (range_1w * 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    # Align weekly pivot levels
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Volume confirmation (20-period average)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Ichimoku signals
        # Price above/both spans = bullish, below/both spans = bearish
        price_above_cloud = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK Cross signals
        tk_cross_bull = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bear = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Chikou confirmation (price vs 26 periods ago)
        chikou_confirm_long = close[i] > chikou_span_aligned[i]
        chikou_confirm_short = close[i] < chikou_span_aligned[i]
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Weekly pivot levels for entry/exit
        # Long: break above R1 with trend alignment
        long_breakout = close[i] > r1_1w_aligned[i]
        # Short: break below S1 with trend alignment
        short_breakout = close[i] < s1_1w_aligned[i]
        
        # Entry conditions
        long_entry = (price_above_cloud and tk_cross_bull and chikou_confirm_long and 
                     long_breakout and vol_confirm)
        short_entry = (price_below_cloud and tk_cross_bear and chikou_confirm_short and 
                      short_breakout and vol_confirm)
        
        # Exit conditions: return to cloud or opposite TK cross
        exit_long = (price_below_cloud or not tk_cross_bull)
        exit_short = (price_above_cloud or not tk_cross_bear)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals