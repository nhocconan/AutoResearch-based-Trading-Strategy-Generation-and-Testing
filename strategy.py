#!/usr/bin/env python3
# 6H_1D_Ichimoku_TK_Cross_Cloud_Filter
# Hypothesis: On 6h timeframe, enter long when Tenkan-sen crosses above Kijun-sen with price above Kumo (cloud) from 1d, and short when Tenkan-sen crosses below Kijun-sen with price below Kumo.
# Uses 1d Ichimoku for trend filter and cloud as dynamic support/resistance. Works in bull via trend continuation, in bear via cloud acting as resistance/support for reversals.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6H_1D_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals, but we need it for completeness
    
    # Determine Kumo (cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For signal at time t, we use Senkou Span values from t-26 (already published)
    shift_kumo = 26
    senkou_span_a_shifted = np.roll(senkou_span_a, shift_kumo)
    senkou_span_b_shifted = np.roll(senkou_span_b, shift_kumo)
    # Set first shift_kumo values to NaN (not yet published)
    senkou_span_a_shifted[:shift_kumo] = np.nan
    senkou_span_b_shifted[:shift_kumo] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_span_a_shifted, senkou_span_b_shifted)
    kumo_bottom = np.minimum(senkou_span_a_shifted, senkou_span_b_shifted)
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate TK cross
        tk_cross_above = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_below = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Price relative to Kumo
        price_above_kumo = close[i] > kumo_top_aligned[i]
        price_below_kumo = close[i] < kumo_bottom_aligned[i]
        
        if position == 0:
            # Enter long: TK cross above + price above Kumo
            if tk_cross_above and price_above_kumo:
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross below + price below Kumo
            elif tk_cross_below and price_below_kumo:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross below OR price drops below Kumo bottom
            if tk_cross_below or not price_above_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross above OR price rises above Kumo top
            if tk_cross_above or not price_below_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals