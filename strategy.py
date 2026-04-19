#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter
# Long when: price > Kumo (cloud), Tenkan > Kijun, and 1d price > 1d EMA50
# Short when: price < Kumo, Tenkan < Kijun, and 1d price < 1d EMA50
# Uses Kumo as dynamic support/resistance, effective in both trending and ranging markets
# Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag
name = "6h_Ichimoku_Cloud_1dEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = (pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Align Ichimoku components (they are already calculated on 6h data, but need proper alignment for plotting)
    # For trading logic, we use the current values directly as they don't involve future data
    tenkan = tenkan_sen.values
    kijun = kijun_sen.values
    span_a = senkou_span_a.values
    span_b = senkou_span_b.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kijun_period, senkou_span_b_period) + 26  # Ensure all components are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(span_a[i]) or 
            np.isnan(span_b[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        # Kumo (cloud) boundaries: Senkou Span A and B
        # The cloud is between span_a and span_b
        upper_kumo = max(span_a[i], span_b[i])
        lower_kumo = min(span_a[i], span_b[i])
        
        # 1d trend filter
        trend_up = close_1d[i//24] > ema_50_aligned[i] if i//24 < len(close_1d) else False  # Simplified alignment check
        trend_down = close_1d[i//24] < ema_50_aligned[i] if i//24 < len(close_1d) else False
        
        # Use aligned 1d EMA for trend
        ema_50_val = ema_50_aligned[i]
        trend_up = close[i] > ema_50_val  # Simplified: use current price vs EMA for trend
        trend_down = close[i] < ema_50_val
        
        # Ichimoku signals
        # Bullish: price above cloud, Tenkan > Kijun
        bullish = price > upper_kumo and tenkan[i] > kijun[i]
        # Bearish: price below cloud, Tenkan < Kijun
        bearish = price < lower_kumo and tenkan[i] < kijun[i]
        
        if position == 0:
            # Enter long if bullish and uptrend
            if bullish and trend_up:
                signals[i] = 0.25
                position = 1
            # Enter short if bearish and downtrend
            elif bearish and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below Kumo or Tenkan < Kijun
            if price < lower_kumo or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above Kumo or Tenkan > Kijun
            if price > upper_kumo or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals