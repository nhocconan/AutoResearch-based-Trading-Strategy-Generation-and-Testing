#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_1dEMA50_Filter_v1
Hypothesis: 6-hour Ichimoku cloud breakout with 1-day EMA50 trend filter. Uses cloud as dynamic support/resistance with TK cross for momentum confirmation. Works in bull via price above cloud with bullish TK cross and 1d uptrend, in bear via price below cloud with bearish TK cross and 1d downtrend. Discrete sizing (0.0, ±0.25) to minimize fee drag. Targets 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku parameters (standard: 9, 26, 52)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26  # kijun period for cloud displacement
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (highest_tenkan + lowest_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (highest_kijun + lowest_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # For alignment, we need to shift back to get current cloud
    # Senkou Span A at current position is calculated from 26 periods ago
    senkou_span_a_lagged = np.roll(senkou_span_a, displacement)
    senkou_span_a_lagged[:displacement] = np.nan
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods plotted 26 periods ahead
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    # Senkou Span B at current position is calculated from 26 periods ago
    senkou_span_b_lagged = np.roll(senkou_span_b, displacement)
    senkou_span_b_lagged[:displacement] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_lagged)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_lagged)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku periods + displacement + 1d EMA
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + displacement + 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        # Cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross conditions
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        # 1d trend filter
        trend_1d_up = close_val > ema_50_1d_aligned[i]
        trend_1d_down = close_val < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price above cloud AND bullish TK cross AND 1d uptrend
            long_signal = (close_val > upper_cloud) and tk_bullish and trend_1d_up
            
            # Short: price below cloud AND bearish TK cross AND 1d downtrend
            short_signal = (close_val < lower_cloud) and tk_bearish and trend_1d_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below cloud OR TK cross turns bearish OR 1d trend turns down
            if (close_val < lower_cloud) or (not tk_bullish) or (not trend_1d_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above cloud OR TK cross turns bullish OR 1d trend turns up
            if (close_val > upper_cloud) or (not tk_bearish) or (not trend_1d_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_1dEMA50_Filter_v1"
timeframe = "6h"
leverage = 1.0