#!/usr/bin/env python3
name = "4h_Ichimoku_Cloud_Reversal_12hTrend"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # 12h trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 26:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Ichimoku on 12h: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52)
    # Tenkan-sen = (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen = (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A = (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B = (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Chikou Span = current close plotted 26 periods back
    chikou = close_12h
    
    # Align all Ichimoku components to 4h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_12h, chikou)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Trend: price above/below cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Reversal signals: Tenkan/Kijun cross + price rejection from cloud
    tenkan_prev = np.roll(tenkan_aligned, 1)
    kijun_prev = np.roll(kijun_aligned, 1)
    tenkan_prev[0] = np.nan
    kijun_prev[0] = np.nan
    
    tk_cross_up = (tenkan_prev <= kijun_prev) & (tenkan_aligned > kijun_aligned)
    tk_cross_down = (tenkan_prev >= kijun_prev) & (tenkan_aligned < kijun_aligned)
    
    # Price rejection from cloud: wick rejection or close rejection
    # Bullish rejection: low touches/cloud penetration then closes back above cloud
    bullish_rejection = (low <= cloud_top) & (close > cloud_bottom) & price_above_cloud
    # Bearish rejection: high touches/cloud penetration then closes back below cloud
    bearish_rejection = (high >= cloud_bottom) & (close < cloud_top) & price_below_cloud
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure Ichimoku calculation is valid
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud AND bullish rejection
            if tk_cross_up[i] and price_above_cloud[i] and bullish_rejection[i]:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud AND bearish rejection
            elif tk_cross_down[i] and price_below_cloud[i] and bearish_rejection[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Tenkan crosses below Kijun OR price falls below cloud
            if tk_cross_down[i] or not price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Tenkan crosses above Kijun OR price rises above cloud
            if tk_cross_up[i] or not price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku Cloud system on 12h timeframe provides robust trend identification
# and dynamic support/resistance. Tenkan/Kijun crossovers signal momentum shifts,
# while price rejection from the cloud confirms the strength of the signal.
# Long when Tenkan crosses above Kijun with price above cloud and bullish rejection.
# Short when Tenkan crosses below Kijun with price below cloud and bearish rejection.
# Works in bull markets (captures uptrend continuations) and bear markets 
# (captures downtrends and bounces). The cloud acts as dynamic support/resistance
# that adapts to volatility, reducing false signals during choppy periods.