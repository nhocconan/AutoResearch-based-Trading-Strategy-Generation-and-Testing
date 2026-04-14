#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation
# Long when price breaks above Kumo (cloud) with volume spike and daily bullish trend
# Short when price breaks below Kumo with volume spike and daily bearish trend
# Exit when price re-enters Kumo
# Uses daily Ichimoku trend to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) with moderate size

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h and daily data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_6h).rolling(window=9, min_periods=9).max().values + 
                  pd.Series(low_6h).rolling(window=9, min_periods=9).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_6h).rolling(window=26, min_periods=26).max().values + 
                 pd.Series(low_6h).rolling(window=26, min_periods=26).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_6h).rolling(window=52, min_periods=52).max().values + 
                      pd.Series(low_6h).rolling(window=52, min_periods=52).min().values) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = np.roll(close, 26)  # Will be handled by alignment
    
    # Daily trend filter using EMA
    close_daily = df_daily['close'].values
    ema_daily = pd.Series(close_daily).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 6h volume average
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=24, min_periods=24).mean().values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    chikou_span_aligned = align_htf_to_ltf(prices, df_6h, chikou_span)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (52 periods for Senkou B)
    start = 52
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_6h_current = volume[i]
        
        # Kumo (Cloud) boundaries - use future values as per Ichimoku definition
        # Senkou Span A and B are already shifted, so we use current values
        upper_kumo = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_kumo = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long setup: price breaks above Kumo with volume spike and daily bullish trend
            if (price > upper_kumo and 
                vol_6h_current > 1.8 * vol_ma_6h_aligned[i] and  # Volume spike
                price > ema_daily_aligned[i]):                  # Price above daily EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Kumo with volume spike and daily bearish trend
            elif (price < lower_kumo and 
                  vol_6h_current > 1.8 * vol_ma_6h_aligned[i] and  # Volume spike
                  price < ema_daily_aligned[i]):                  # Price below daily EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price re-enters Kumo (falls below upper Kumo)
            if price < upper_kumo:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price re-enters Kumo (rises above lower Kumo)
            if price > lower_kumo:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Ichimoku_Kumo_Breakout_Volume_DailyTrend"
timeframe = "6h"
leverage = 1.0