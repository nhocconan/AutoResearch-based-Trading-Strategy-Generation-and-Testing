#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with weekly trend filter and volume confirmation
# Uses 1w Ichimoku for primary trend direction (avoids counter-trend trades)
# 6h Tenkan/Kijun cross for entry timing with price outside cloud for confirmation
# Volume spike (>1.5x 24-bar average) ensures institutional participation
# Designed for low trade frequency (15-35/year) to minimize fee drag
# Works in bull/bear: trend filter captures major moves, cloud acts as dynamic support/resistance

name = "6h_Ichimoku_1wTrend_Volume_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 52 or len(df_1d) < 26:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_9_1w + low_9_1w) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_26_1w + low_26_1w) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods
    high_52_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = ((high_52_1w + low_52_1w) / 2)
    
    # Calculate 1d trend filter (EMA50)
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume spike filter (>1.5x 24-bar average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    # Align HTF indicators to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        lower_cloud = np.minimum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        
        if position == 0:
            # Long entry: Tenkan > Kijun AND price above cloud AND uptrend (price > EMA50) AND volume spike
            if (tenkan_1w_aligned[i] > kijun_1w_aligned[i] and 
                close[i] > upper_cloud and 
                close[i] > ema50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Tenkan < Kijun AND price below cloud AND downtrend (price < EMA50) AND volume spike
            elif (tenkan_1w_aligned[i] < kijun_1w_aligned[i] and 
                  close[i] < lower_cloud and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below cloud OR Tenkan < Kijun
            if close[i] < lower_cloud or tenkan_1w_aligned[i] < kijun_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above cloud OR Tenkan > Kijun
            if close[i] > upper_cloud or tenkan_1w_aligned[i] > kijun_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals