#!/usr/bin/env python3
"""
6h_1d_Ichimoku_Cloud_Breakout_v1
Hypothesis: Use Ichimoku Cloud from daily timeframe as major support/resistance. 
Go long when price breaks above the cloud with bullish TK cross, short when breaks below with bearish TK cross.
Works in bull/bear markets by using cloud as dynamic S/R and TK cross for momentum confirmation.
Targets 20-40 trades/year on 6f timeframe by requiring both cloud break and TK alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # For Ichimoku signals, we don't actually plot it back, we compare current price to it
    # But for our use, we'll use the current close vs the close 26 periods ago
    chikou_span = np.roll(close_1d, 26)  # Shifted back 26 periods
    chikou_span[:26] = np.nan  # First 26 values invalid
    
    # Align all Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(span_a_6h[i]) or np.isnan(span_b_6h[i]) or 
            np.isnan(chikou_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(span_a_6h[i], span_b_6h[i])
        lower_cloud = np.minimum(span_a_6h[i], span_b_6h[i])
        
        # TK Cross conditions
        tk_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        price_in_cloud = (close[i] >= lower_cloud) and (close[i] <= upper_cloud)
        
        # Chikou confirmation: today's price vs price 26 periods ago
        # Chikou is current close plotted 26 periods back, so we compare it to price 26 periods ago
        chikou_confirm_bullish = chikou_6h[i] > close_1d[max(0, i-26)] if i >= 26 else False
        chikou_confirm_bearish = chikou_6h[i] < close_1d[max(0, i-26)] if i >= 26 else False
        
        # Entry conditions:
        # Long: price breaks above cloud + bullish TK cross + Chikou confirmation
        long_entry = price_above_cloud and tk_bullish and chikou_confirm_bullish and not price_in_cloud
        
        # Short: price breaks below cloud + bearish TK cross + Chikou confirmation
        short_entry = price_below_cloud and tk_bearish and chikou_confirm_bearish and not price_in_cloud
        
        # Exit conditions:
        # Long exit: price returns to cloud or TK cross turns bearish
        long_exit = price_in_cloud or (not tk_bullish)
        
        # Short exit: price returns to cloud or TK cross turns bullish
        short_exit = price_in_cloud or (not tk_bearish)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals