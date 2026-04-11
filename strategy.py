#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_trend_v1
# Strategy: 6s Ichimoku Cloud with 1d trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Ichimoku Cloud provides dynamic support/resistance and trend direction. Combined with 1d trend filter and volume confirmation, it captures strong trends while avoiding false signals in choppy markets. Designed for low trade frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df_1d['high'].rolling(window=9, min_periods=9).max().values
    period9_low = df_1d['low'].rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df_1d['high'].rolling(window=26, min_periods=26).max().values
    period26_low = df_1d['low'].rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = df_1d['high'].rolling(window=52, min_periods=52).max().values
    period52_low = df_1d['low'].rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = df_1d['close'].shift(26).values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    # 1d ADX for trend strength filter
    # Calculate +DM, -DM, TR
    high_diff = df_1d['high'].diff()
    low_diff = df_1d['low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    plus_di_14 = np.where(atr_14 != 0, 100 * plus_dm_14 / atr_14, 0)
    minus_di_14 = np.where(atr_14 != 0, 100 * minus_dm_14 / atr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 0)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i]) or np.isnan(adx_14_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # ADX trend strength: ADX > 25 indicates strong trend
        strong_trend = adx_14_aligned[i] > 25
        
        # Ichimoku signals
        # Price above cloud (bullish)
        price_above_cloud = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        # Price below cloud (bearish)
        price_below_cloud = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        # TK Cross: Tenkan-sen crosses above Kijun-sen (bullish)
        tk_cross_bullish = (tenkan_sen_aligned[i] > kijun_sen_aligned[i]) and (tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
        # TK Cross: Tenkan-sen crosses below Kijun-sen (bearish)
        tk_cross_bearish = (tenkan_sen_aligned[i] < kijun_sen_aligned[i]) and (tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
        
        # Entry conditions
        # Long: Price above cloud AND TK cross bullish AND strong trend AND volume confirmation
        if price_above_cloud and tk_cross_bullish and strong_trend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price below cloud AND TK cross bearish AND strong trend AND volume confirmation
        elif price_below_cloud and tk_cross_bearish and strong_trend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses opposite TK line or price enters cloud
        elif position == 1 and (tk_cross_bearish or close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tk_cross_bullish or close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals