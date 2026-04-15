#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + TK Cross with 1d trend filter
# Uses Ichimoku system (Tenkan/Kijun/Senkou) on 6h for entry signals,
# filtered by 1d EMA50 trend to avoid counter-trend trades.
# Works in bull/bear by only taking trades in direction of higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_6h).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_6h).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_6h).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_6h).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_6h).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_6h).rolling(window=52, min_periods=52).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 22 periods behind
    chikou_span = pd.Series(close_6h).shift(22).values
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    chikou_span_aligned = align_htf_to_ltf(prices, df_6h, chikou_span)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Long entry: TK cross bullish + price above cloud + price above 1d EMA50
        if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and  # TK cross bullish
            close[i] > cloud_top and                          # Price above cloud
            close[i] > ema50_1d_aligned[i] and                # Price above 1d EMA50
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: TK cross bearish + price below cloud + price below 1d EMA50
        elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and  # TK cross bearish
              close[i] < cloud_bottom and                       # Price below cloud
              close[i] < ema50_1d_aligned[i] and                # Price below 1d EMA50
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: TK cross in opposite direction or price enters cloud
        elif position == 1 and (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                                close[i] < cloud_top):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                                 close[i] > cloud_bottom):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Trend_Filter"
timeframe = "6h"
leverage = 1.0