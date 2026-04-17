#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using Ichimoku Cloud (Tenkan/Kijun from 1d, Senkou Span A/B from 1d).
- Long when price above cloud, Tenkan > Kijun, and volume > 1.5x 20-period volume MA
- Short when price below cloud, Tenkan < Kijun, and volume > 1.5x 20-period volume MA
- Exit when price crosses back into cloud or Tenkan/Kijun cross reverses
- Uses cloud as dynamic support/resistance, effective in both trending and ranging markets
- Fixed position size 0.25 to manage drawdown
- Designed for 6h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period_kijun)  # shifted 26 periods ahead
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(period_kijun)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 52  # warmup for Ichimoku calculations
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Look for signals with volume confirmation
            # Long: price above cloud, Tenkan > Kijun, volume spike
            if price > cloud_top and tenkan > kijun and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, Tenkan < Kijun, volume spike
            elif price < cloud_bottom and tenkan < kijun and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below cloud or Tenkan/Kijun cross down
            if price < cloud_top or tenkan < kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above cloud or Tenkan/Kijun cross up
            if price > cloud_bottom or tenkan > kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Volume"
timeframe = "6h"
leverage = 1.0