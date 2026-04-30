#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Ichimoku Cloud (Tenkan/Kijun) with 1d trend filter
# Ichimoku Cloud acts as dynamic support/resistance; TK cross with cloud filter captures momentum
# aligned with higher timeframe trend. Works in bull/bear markets by requiring
# 1d EMA50 alignment to avoid counter-trend trades. Target: 50-150 total trades over 4 years.

name = "6h_Ichimoku_TK_Cross_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Ichimoku calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # need 26*2 for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (wait for completed 12h bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Ichimoku calculations
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_tenkan = tenkan_aligned[i]
        curr_kijun = kijun_aligned[i]
        curr_senkou_a = senkou_a_aligned[i]
        curr_senkou_b = senkou_b_aligned[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        if position == 0:  # Flat - look for new entries
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_cross_up = curr_tenkan > curr_kijun and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_cross_down = curr_tenkan < curr_kijun and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            
            if tk_cross_up and curr_close > cloud_top and curr_close > curr_ema_1d:
                # Bullish: price above cloud and above 1d EMA50
                signals[i] = 0.25
                position = 1
            elif tk_cross_down and curr_close < cloud_bottom and curr_close < curr_ema_1d:
                # Bearish: price below cloud and below 1d EMA50
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun OR price drops below cloud bottom
            tk_cross_down = curr_tenkan < curr_kijun and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            if tk_cross_down or curr_close < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun OR price rises above cloud top
            tk_cross_up = curr_tenkan > curr_kijun and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            if tk_cross_up or curr_close > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals