#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 6h and 1d data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_6h) < 26 or len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Rolling max/min for Tenkan-sen
    high_roll_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_roll_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_roll_tenkan + low_roll_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    high_roll_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_roll_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_roll_kijun + low_roll_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    high_roll_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_roll_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (high_roll_senkou_b + low_roll_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe (no shift as they are leading spans)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 6h volume spike: > 2x 20-period average
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_6h = volume > 2 * vol_ma_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 34)  # Wait for Senkou Span B and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: Tenkan > Kijun, price above cloud, bullish trend (price > EMA34), volume spike
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                close[i] > cloud_top and 
                close[i] > ema34_1d_aligned[i] and 
                vol_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun, price below cloud, bearish trend (price < EMA34), volume spike
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < cloud_bottom and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_spike_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below cloud or Tenkan < Kijun
            if close[i] < cloud_bottom or tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above cloud or Tenkan > Kijun
            if close[i] > cloud_top or tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku cloud acts as dynamic support/resistance. TK cross confirms momentum.
# In bull markets: price stays above cloud, TK cross up signals continuation.
# In bear markets: price stays below cloud, TK cross down signals continuation.
# Volume spike filters for institutional participation. 1d EMA34 ensures alignment with daily trend.
# Target: 15-25 trades/year per symbol to minimize fee drift. Use 0.25 position size for risk control.