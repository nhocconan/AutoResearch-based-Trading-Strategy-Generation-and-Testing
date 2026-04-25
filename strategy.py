#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1wCloud_Filter_1dTrend
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun cross with weekly cloud filter and daily trend confirmation.
Tenkan-sen (9-period) and Kijun-sen (26-period) cross provides momentum signals.
Weekly cloud (Senkou Span A/B from 26 periods ago) acts as major support/resistance filter.
Daily EMA50 trend filter ensures alignment with intermediate-term direction.
Designed for 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
Works in bull markets via TK cross above cloud and in bear markets via TK cross below cloud.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1w data for Ichimoku cloud (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku calculations on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1w['high'].values).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1w['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1w['high'].values).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1w['low'].values).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = pd.Series(df_1w['high'].values).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1w['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (completed weekly bar)
    # Tenkan/Kijun need no additional delay as they are based on completed weekly bar
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    
    # Senkou Span A/B need 26-bar delay because they are plotted 26 periods ahead
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b, additional_delay_bars=26)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly calculations (52) + 1d EMA (50)
    start_idx = max(52, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Look for entry signals - require: TK cross + price relative to cloud + daily trend alignment
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
            tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
            
            price_above_cloud = curr_close > upper_cloud
            price_below_cloud = curr_close < lower_cloud
            
            # Trend filter: price must be on correct side of daily EMA50
            long_trend = curr_close > ema_50_1d_aligned[i]
            short_trend = curr_close < ema_50_1d_aligned[i]
            
            # Long: TK bullish cross AND price above cloud AND daily uptrend
            long_entry = tk_bullish and price_above_cloud and long_trend
            
            # Short: TK bearish cross AND price below cloud AND daily downtrend
            short_entry = tk_bearish and price_below_cloud and short_trend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when TK turns bearish OR price falls below cloud
            if tenkan_aligned[i] < kijun_aligned[i] or curr_close < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when TK turns bullish OR price rises above cloud
            if tenkan_aligned[i] > kijun_aligned[i] or curr_close > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wCloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0