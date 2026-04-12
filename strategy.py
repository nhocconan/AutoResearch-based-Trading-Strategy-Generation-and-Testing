#!/usr/bin/env python3
"""
6h_1d_ICHIMOKU_CLOUD_FILTER
Hypothesis: On 6h timeframe, enter long when Tenkan-sen crosses above Kijun-sen and price is above the Kumo (cloud) from daily timeframe, with volume confirmation. Enter short when Tenkan-sen crosses below Kijun-sen and price is below the Kumo with volume confirmation. Uses daily Ichimoku cloud for trend filter and 6h Tenkan/Kijun cross for entry timing. Volume filter ensures momentum confirmation. Target: 15-25 trades per year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ICHIMOKU_CLOUD_FILTER"
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
    
    # === DAILY INDICATORS: Ichimoku Cloud (9, 26, 52) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted 26 periods ahead
    senkou_span_a_shifted = np.roll(senkou_span_a, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b, 26)
    # First 26 values are invalid (no future data)
    senkou_span_a_shifted[:26] = np.nan
    senkou_span_b_shifted[:26] = np.nan
    
    # Align to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a_shifted)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b_shifted)
    
    # Volume filter: volume > 1.5 * average volume of prior 20 periods
    vol_ma = np.zeros_like(volume)
    vol_ma[20] = np.mean(volume[0:20])
    for i in range(21, len(volume)):
        vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Kumo top and bottom
        kumo_top = np.maximum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        kumo_bottom = np.minimum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        # Tenkan/Kijun cross
        tk_cross_up = tenkan_sen_6h[i] > kijun_sen_6h[i] and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
        tk_cross_down = tenkan_sen_6h[i] < kijun_sen_6h[i] and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]
        
        # Price relative to Kumo
        price_above_kumo = close[i] > kumo_top
        price_below_kumo = close[i] < kumo_bottom
        
        # Entry conditions with volume confirmation
        long_entry = tk_cross_up and price_above_kumo and volume_filter[i]
        short_entry = tk_cross_down and price_below_kumo and volume_filter[i]
        
        # Exit conditions: opposite TK cross or price re-enters Kumo
        exit_long = tk_cross_down or not price_above_kumo
        exit_short = tk_cross_up or not price_below_kumo
        
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