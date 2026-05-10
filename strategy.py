#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_1dTrend_Volume
# Hypothesis: 6h Ichimoku Tenkan/Kijun cross with 1d Kumo cloud filter and volume confirmation.
# The Ichimoku cloud acts as dynamic support/resistance, with the 1d cloud providing higher timeframe context.
# In trending markets, price stays above/below cloud with TK cross signaling momentum.
# In ranging markets, price oscillates within cloud, reducing false signals.
# Volume filter ensures breakouts have conviction. Targets 15-30 trades/year to minimize fee drag.
# Works in bull/bear by trading with higher timeframe trend (cloud color) and momentum (TK cross).

name = "6h_Ichimoku_Cloud_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # 1d data for Ichimoku cloud (higher timeframe context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo (cloud) boundaries - shifted forward by 26 periods
    senkou_span_a_shifted = np.roll(senkou_span_a, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b, 26)
    # Fill first 26 values with NaN to avoid look-ahead
    senkou_span_a_shifted[:26] = np.nan
    senkou_span_b_shifted[:26] = np.nan
    
    # Align Ichimoku components to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a_shifted)
    span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b_shifted)
    
    # Determine cloud color: green if span_a > span_b (bullish), red if span_a < span_b (bearish)
    cloud_green = span_a_6h > span_b_6h
    cloud_red = span_a_6h < span_b_6h
    
    # 6x multiplier for volume average (more conservative for 6h timeframe)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=6 * 20, min_periods=6 * 20).mean().values  # 120 periods
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 6 * 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(span_a_6h[i]) or np.isnan(span_b_6h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        # Determine cloud boundaries (top and bottom of cloud)
        cloud_top = np.maximum(span_a_6h[i], span_b_6h[i])
        cloud_bottom = np.minimum(span_a_6h[i], span_b_6h[i])
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + green cloud + volume
            tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
            price_above_cloud = close[i] > cloud_top
            if tk_cross_bullish and price_above_cloud and cloud_green[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud + red cloud + volume
            elif (tenkan_6h[i] < kijun_6h[i] and
                  close[i] < cloud_bottom and
                  cloud_red[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TK cross bearish OR price drops below cloud bottom
            tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
            price_below_cloud = close[i] < cloud_bottom
            if tk_cross_bearish or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TK cross bullish OR price rises above cloud top
            tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
            price_above_cloud = close[i] > cloud_top
            if tk_cross_bullish or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals