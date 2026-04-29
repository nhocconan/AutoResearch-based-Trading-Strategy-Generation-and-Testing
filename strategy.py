#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + TK Cross + Weekly Trend Filter
# Long when: Tenkan > Kijun (TK Cross bullish) AND price > Kumo cloud (from 1d) AND weekly EMA50 uptrend
# Short when: Tenkan < Kijun (TK Cross bearish) AND price < Kumo cloud AND weekly EMA50 downtrend
# Exit when: TK Cross reverses OR price re-enters cloud
# Uses Ichimoku for trend/momentum, weekly EMA for higher-timeframe bias, cloud for dynamic support/resistance
# Designed to capture sustained moves in both bull and bear markets via trend alignment and volatility-based cloud

name = "6h_Ichimoku_TK_Cross_1dCloud_1wEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df_1d['high'].rolling(window=9, min_periods=9).max().values
    period9_low = df_1d['low'].rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df_1d['high'].rolling(window=26, min_periods=26).max().values
    period26_low = df_1d['low'].rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = df_1d['high'].rolling(window=52, min_periods=52).max().values
    period52_low = df_1d['low'].rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Align Ichimoku components to 6h timeframe (they are already forward-shifted by 26 in calculation)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 50)  # Ichimoku and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_tenkan = tenkan_6h[i]
        curr_kijun = kijun_6h[i]
        curr_senkou_a = senkou_a_6h[i]
        curr_senkou_b = senkou_b_6h[i]
        curr_ema50w = ema_50_1w_aligned[i]
        curr_close = close[i]
        
        # Determine cloud top and bottom
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: TK Cross bearish OR price re-enters cloud
            if curr_tenkan < curr_kijun or (curr_close > cloud_bottom and curr_close < cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK Cross bullish OR price re-enters cloud
            if curr_tenkan > curr_kijun or (curr_close > cloud_bottom and curr_close < cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when: TK Cross bullish AND price above cloud AND weekly EMA50 uptrend
            if curr_tenkan > curr_kijun and curr_close > cloud_top and curr_close > curr_ema50w:
                signals[i] = 0.25
                position = 1
            # Short when: TK Cross bearish AND price below cloud AND weekly EMA50 downtrend
            elif curr_tenkan < curr_kijun and curr_close < cloud_bottom and curr_close < curr_ema50w:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals