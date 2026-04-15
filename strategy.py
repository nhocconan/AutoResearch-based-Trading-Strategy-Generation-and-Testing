#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud strategy with 1d trend filter
# Uses Tenkan-sen (9-period) and Kijun-sen (26-period) cross for momentum signals
# Filters trades by Senkou Span (cloud) from 1d timeframe to avoid counter-trend trades
# Works in bull markets (price above cloud) and bear markets (price below cloud)
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag
# Target: 50-150 total trades over 4 years

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_span_b = (high_senkou_b + low_senkou_b) / 2
    
    # Get 1d data for trend filter (cloud)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku cloud components
    high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_sen_1d = (high_tenkan_1d + low_tenkan_1d) / 2
    
    high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_sen_1d = (high_kijun_1d + low_kijun_1d) / 2
    
    senkou_span_a_1d = (tenkan_sen_1d + kijun_sen_1d) / 2
    
    high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_span_b_1d = (high_senkou_b_1d + low_senkou_b_1d) / 2
    
    # Align 1d cloud to 6s timeframe
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    signals = np.zeros(n)
    
    for i in range(52, n):  # Start after warmup for Senkou Span B
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i])):
            continue
        
        # Determine cloud top and bottom for 1d
        cloud_top = max(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        cloud_bottom = min(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        
        # Bullish TK cross: Tenkan-sen crosses above Kijun-sen
        tk_cross_bullish = (tenkan_sen[i] > kijun_sen[i]) and (tenkan_sen[i-1] <= kijun_sen[i-1])
        # Bearish TK cross: Tenkan-sen crosses below Kijun-sen
        tk_cross_bearish = (tenkan_sen[i] < kijun_sen[i]) and (tenkan_sen[i-1] >= kijun_sen[i-1])
        
        # Long: Bullish TK cross + price above 1d cloud
        if tk_cross_bullish and close[i] > cloud_top:
            signals[i] = 0.25
        
        # Short: Bearish TK cross + price below 1d cloud
        elif tk_cross_bearish and close[i] < cloud_bottom:
            signals[i] = -0.25
        
        # Exit: TK cross in opposite direction
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and tk_cross_bearish) or
               (signals[i-1] == -0.25 and tk_cross_bullish))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_IchimokuCloud_TKCross_1dFilter"
timeframe = "6h"
leverage = 1.0