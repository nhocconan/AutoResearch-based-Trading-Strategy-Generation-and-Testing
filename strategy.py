#!/usr/bin/env python3
# 6h_ichimoku_trend_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d timeframe for trend direction and 6h TK cross for entry timing.
# Long when price is above 1d Ichimoku cloud and 6h Tenkan-sen crosses above Kijun-sen.
# Short when price is below 1d Ichimoku cloud and 6h Tenkan-sen crosses below Kijun-sen.
# Exit when price crosses the opposite 6h TK line or re-enters the cloud.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Ichimoku works in both bull and bear markets by filtering trend direction from higher timeframe.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_trend_v1"
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
    
    # Get daily data for Ichimoku cloud (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_10 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_10 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_10 + low_10) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (high_52 + low_52) / 2.0
    
    # Align Ichimoku components to 6h timeframe (with proper delay for completed daily bars)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6h TK cross for entry timing
    # Tenkan-sen (Conversion Line) on 6h
    period_tenkan_6h = 9
    high_10_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    low_10_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_sen_6h = (high_10_6h + low_10_6h) / 2.0
    
    # Kijun-sen (Base Line) on 6h
    period_kijun_6h = 26
    high_26_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    low_26_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_sen_6h = (high_26_6h + low_26_6h) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Check if price is above or below cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        if position == 1:  # Long position
            # Exit: Price crosses below 6h Kijun-sen or re-enters cloud
            if (tenkan_sen_6h[i] < kijun_sen_6h[i]) or (close[i] < upper_cloud and close[i] > lower_cloud):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above 6h Kijun-sen or re-enters cloud
            if (tenkan_sen_6h[i] > kijun_sen_6h[i]) or (close[i] < upper_cloud and close[i] > lower_cloud):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for TK cross with cloud filter
            bullish_cross = tenkan_sen_6h[i] > kijun_sen_6h[i] and price_above_cloud
            bearish_cross = tenkan_sen_6h[i] < kijun_sen_6h[i] and price_below_cloud
            
            if bullish_cross:
                position = 1
                signals[i] = 0.25
            elif bearish_cross:
                position = -1
                signals[i] = -0.25
    
    return signals