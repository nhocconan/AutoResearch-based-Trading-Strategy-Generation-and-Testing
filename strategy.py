#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d Trend Filter
# Uses Ichimoku (Tenkan/Kijun/Senkou) on 6h for entry signals and 1d trend (price vs EMA50) as filter.
# Long when Tenkan > Kijun and price above Kumo (cloud) and 1d uptrend.
# Short when Tenkan < Kijun and price below Kumo and 1d downtrend.
# Ichimoku works well in both trending and ranging markets; 1d filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    kumo_shift = 26
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() +
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kumo_shift)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() +
                      pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(kumo_shift)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, prices, tenkan_sen.values)  # same timeframe
    kijun_sen_aligned = align_htf_to_ltf(prices, prices, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, prices, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, prices, senkou_span_b.values)
    
    # Align 1d EMA50 to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            continue
        
        # Determine Kumo (cloud) boundaries
        upper_kumo = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_kumo = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Long entry: Tenkan > Kijun, price above Kumo, and 1d uptrend (price > EMA50)
        if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
            close[i] > upper_kumo and
            close[i] > ema_50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Tenkan < Kijun, price below Kumo, and 1d downtrend (price < EMA50)
        elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
              close[i] < lower_kumo and
              close[i] < ema_50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Ichimoku cross or price crosses EMA50 (trend change)
        elif position == 1 and (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0