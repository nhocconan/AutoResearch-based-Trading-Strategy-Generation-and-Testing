#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when price breaks above 6h Ichimoku cloud AND 1d close > 1d EMA50 AND volume > 1.5 * 20-bar average volume
# Short when price breaks below 6h Ichimoku cloud AND 1d close < 1d EMA50 AND volume > 1.5 * 20-bar average volume
# Exit when price re-enters the 6h Ichimoku cloud
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe
# Ichimoku cloud provides dynamic support/resistance that adapts to volatility
# 1d EMA50 filters for higher timeframe trend alignment
# Volume confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by requiring trend alignment and structure breaks

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_Volume_v1"
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
    
    # Calculate 6h Ichimoku and 1d EMA50 ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_6h) < 52 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    high_52 = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2.0
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed bars)
    # Ichimoku components need to be shifted forward by 26 periods for cloud
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long breakout: price > upper cloud AND uptrend AND volume spike
            if close[i] > upper_cloud and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower cloud AND downtrend AND volume spike
            elif close[i] < lower_cloud and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters cloud (falls below upper cloud)
            if close[i] <= upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters cloud (rises above lower cloud)
            if close[i] >= lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals