#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku cloud for trend filter and 6h Williams %R for entry timing
# Ichimoku cloud from 1d provides major trend support/resistance (bullish when price above cloud, bearish when below)
# Williams %R on 6h timeframe identifies overbought/oversold conditions within the 1d trend
# Long: price above 1d cloud AND Williams %R < -80 (oversold pullback in uptrend)
# Short: price below 1d cloud AND Williams %R > -20 (overbought pullback in downtrend)
# Works in bull/bear: follows major 1d trend while picking entries on 6h pullbacks
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_ichimoku_williamsr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Ichimoku calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h timeframe (default align_htf_to_ltf handles completed bar delay)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6h Williams %R
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period14_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    period14_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((period14_high - close) / (period14_high - period14_low)) * -100
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Determine Ichimoku trend: price above/below cloud
        # Cloud top = max(Span A, Span B), Cloud bottom = min(Span A, Span B)
        cloud_top = np.maximum(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = np.minimum(span_a_aligned[i], span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 1:  # Long position
            # Exit when price breaks below cloud or Williams %R shows overbought
            if price_below_cloud or williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price breaks above cloud or Williams %R shows oversold
            if price_above_cloud or williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter on pullbacks in the direction of 1d Ichimoku trend
            # Long: price above cloud AND Williams %R < -80 (oversold)
            # Short: price below cloud AND Williams %R > -20 (overbought)
            if price_above_cloud and williams_r[i] < -80:
                position = 1
                signals[i] = 0.25
            elif price_below_cloud and williams_r[i] > -20:
                position = -1
                signals[i] = -0.25
    
    return signals