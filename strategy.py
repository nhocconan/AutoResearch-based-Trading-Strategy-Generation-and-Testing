#!/usr/bin/env python3
"""
6h_ichimoku_1d_trend_v1
Hypothesis: On 6-hour timeframe, use Ichimoku cloud from 1-day timeframe for trend direction and Tenkan/Kijun cross for entries.
Long when price is above Kumo (cloud) and Tenkan crosses above Kijun.
Short when price is below Kumo and Tenkan crosses below Kijun.
Exit when price crosses back into the Kumo or Tenkan/Kijun reverses.
Designed for 15-30 trades/year to avoid overtrading while capturing trends in both bull and bear markets.
Ichimoku adapts to volatility and provides dynamic support/resistance, making it effective across regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(1)  # Shifted for cloud projection
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(1)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Determine Kumo (cloud) boundaries
    kumou_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumou_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Determine trend: price above/below cloud
    price_above_kumo = close > kumou_top
    price_below_kumo = close < kumou_bottom
    
    # Tenkan/Kijun cross signals
    tenkan_kijun_cross_up = (tenkan_sen_aligned > kijun_sen_aligned) & (tenkan_sen_aligned <= kijun_sen_aligned.shift(1))
    tenkan_kijun_cross_down = (tenkan_sen_aligned < kijun_sen_aligned) & (tenkan_sen_aligned >= kijun_sen_aligned.shift(1))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Senkou B lookback
        # Skip if data not available
        if (np.isnan(kumou_top[i]) or np.isnan(kumou_bottom[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i])):
            signals[i] = 0.0
            continue
            
        if position == 1:  # Long position
            # Exit: price crosses below Kumo OR Tenkan crosses below Kijun
            if close[i] <= kumou_top[i] or tenkan_kijun_cross_down[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Kumo OR Tenkan crosses above Kijun
            if close[i] >= kumou_bottom[i] or tenkan_kijun_cross_up[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above Kumo AND Tenkan crosses above Kijun
            if price_above_kumo[i] and tenkan_kijun_cross_up[i]:
                position = 1
                signals[i] = 0.25
            # Short: price below Kumo AND Tenkan crosses below Kijun
            elif price_below_kumo[i] and tenkan_kijun_cross_down[i]:
                position = -1
                signals[i] = -0.25
    
    return signals