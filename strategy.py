#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1w trend filter
# Ichimoku provides a complete trend system: TK cross for momentum, cloud for support/resistance.
# We use weekly trend (price above/below weekly Kumo) to filter signals, ensuring we only trade
# in the direction of the higher timeframe trend. This avoids counter-trend whipsaws in both
# bull and bear markets. The cloud acts as dynamic support/resistance, reducing false breakouts.
# Targets 20-30 trades per year (~80-120 total over 4 years) with clear entry/exit rules.

name = "6h_Ichimoku_Cloud_1wTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters (standard)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Tenkan-sen and Kijun-sen
    tenkan_sen_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    
    # Weekly Senkou Span A and B
    senkou_span_a_1w = (tenkan_sen_1w + kijun_sen_1w) / 2
    senkou_span_b_1w = (pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                        pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2
    
    # Weekly Kumo (cloud) edges
    kumo_top_1w = np.maximum(senkou_span_a_1w, senkou_span_b_1w)
    kumo_bottom_1w = np.minimum(senkou_span_a_1w, senkou_span_b_1w)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1w, tenkan_sen.values)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1w, kijun_sen.values)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1w, senkou_span_a.values)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1w, senkou_span_b.values)
    
    # Align weekly Kumo edges to 6h
    kumo_top_6h = align_htf_to_ltf(prices, df_1w, kumo_top_1w)
    kumo_bottom_6h = align_htf_to_ltf(prices, df_1w, kumo_bottom_1w)
    
    # Determine trend: price above/below weekly Kumo
    # For 6h bar, we use the aligned weekly Kumo values
    price_above_kumo = close > kumo_top_6h
    price_below_kumo = close < kumo_bottom_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period)  # Ensure all components ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or
            np.isnan(kumo_top_6h[i]) or np.isnan(kumo_bottom_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate cloud boundaries for current 6h bar
        senkou_span_a_current = senkou_span_a_6h[i]
        senkou_span_b_current = senkou_span_b_6h[i]
        kumo_top = max(senkou_span_a_current, senkou_span_b_current)
        kumo_bottom = min(senkou_span_a_current, senkou_span_b_current)
        
        if position == 0:
            # Enter long: TK cross bullish + price above weekly Kumo
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] and 
                tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1] and  # Fresh cross
                price_above_kumo[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross bearish + price below weekly Kumo
            elif (tenkan_sen_6h[i] < kijun_sen_6h[i] and 
                  tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1] and  # Fresh cross
                  price_below_kumo[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud
            if (tenkan_sen_6h[i] < kijun_sen_6h[i] and 
                tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]) or close[i] < kumo_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] and 
                tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]) or close[i] > kumo_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals