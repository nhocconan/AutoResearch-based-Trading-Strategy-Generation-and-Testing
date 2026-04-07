#!/usr/bin/env python3
"""
6h_ichimoku_cloud_1d_trend_v1
Hypothesis: Ichimoku Cloud on daily timeframe provides strong trend direction and support/resistance.
On 6h timeframe, enter long when price is above cloud with bullish TK cross, 
enter short when price is below cloud with bearish TK cross.
Works in trending markets (both bull/bear) by filtering with cloud color and TK cross.
Avoids range-bound markets via cloud thickness filter.
Target: 15-30 trades/year on 6h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (standard periods: 9, 26, 52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2)
    
    # Chikou Span (Lagging Span): current close shifted 26 periods behind
    # Not used for signals but calculated for completeness
    
    # Align Ichimoku components to 6h timeframe (with proper shift for forward-looking components)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if data not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Cloud boundaries (Senkou Span A and B form the cloud)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Cloud thickness filter: avoid thin clouds (weak support/resistance)
        cloud_thickness = cloud_top - cloud_bottom
        if i >= 26:  # Need enough history for thickness calculation
            # Calculate average cloud thickness over last 26 periods
            start_idx = max(0, i - 25)
            thickness_values = []
            for j in range(start_idx, i + 1):
                if not (np.isnan(senkou_span_a_aligned[j]) or np.isnan(senkou_span_b_aligned[j])):
                    t_top = max(senkou_span_a_aligned[j], senkou_span_b_aligned[j])
                    t_bottom = min(senkou_span_a_aligned[j], senkou_span_b_aligned[j])
                    thickness_values.append(t_top - t_bottom)
            if thickness_values:
                avg_thickness = np.mean(thickness_values[-26:]) if len(thickness_values) >= 26 else np.mean(thickness_values)
                # Skip if cloud is too thin (less than 0.5% of price)
                if avg_thickness < 0.005 * close[i]:
                    signals[i] = 0.0
                    continue
        
        # Determine cloud color (green = bullish, red = bearish)
        # Green cloud when Span A > Span B, red when Span A < Span B
        is_bullish_cloud = span_a > span_b
        
        if position == 1:  # Long position
            # Exit: price drops below cloud or TK cross turns bearish
            if close[i] < cloud_bottom or tenkan < kijun:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud or TK cross turns bullish
            if close[i] > cloud_top or tenkan > kijun:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above cloud, bullish cloud, and bullish TK cross
            if close[i] > cloud_top and is_bullish_cloud and tenkan > kijun:
                position = 1
                signals[i] = 0.25
            # Short: price below cloud, bearish cloud, and bearish TK cross
            elif close[i] < cloud_bottom and not is_bullish_cloud and tenkan < kijun:
                position = -1
                signals[i] = -0.25
    
    return signals