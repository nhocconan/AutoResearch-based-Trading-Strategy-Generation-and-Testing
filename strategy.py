#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_12hTrend_VolumeSpike
Hypothesis: On 6h timeframe, use Ichimoku TK cross (Tenkan/Kijun) for entry signals with 12h trend filter (price above/below Kumo cloud) and volume confirmation (>2.0x 20-period average). This combines momentum (TK cross) with trend filter (cloud) and volume to reduce false signals. The 6h timeframe targets 12-37 trades/year with discrete sizing (0.25) to manage fees and drawdown in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Ichimoku calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # Need at least 52 periods for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate 12h OHLC for Ichimoku
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_12h).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_12h).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_12h).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_12h).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_12h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_12h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Current price vs Kumo (cloud) for trend filter
    # Price above cloud = bullish, Price below cloud = bearish
    price_above_cloud = close_12h > np.maximum(senkou_span_a, senkou_span_b)
    price_below_cloud = close_12h < np.minimum(senkou_span_a, senkou_span_b)
    
    # TK cross signals
    # Bullish TK cross: Tenkan crosses above Kijun
    tk_bullish = (tenkan_sen > kijun_sen) & (tenkan_sen.shift(1) <= kijun_sen.shift(1))
    # Bearish TK cross: Tenkan crosses below Kijun
    tk_bearish = (tenkan_sen < kijun_sen) & (tenkan_sen.shift(1) >= kijun_sen.shift(1))
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b.values)
    price_above_cloud_aligned = align_htf_to_ltf(prices, df_12h, price_above_cloud.values)
    price_below_cloud_aligned = align_htf_to_ltf(prices, df_12h, price_below_cloud.values)
    tk_bullish_aligned = align_htf_to_ltf(prices, df_12h, tk_bullish.values)
    tk_bearish_aligned = align_htf_to_ltf(prices, df_12h, tk_bearish.values)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku (52 periods) + volume MA warmup
    start_idx = max(52, 20)  # 52 for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(price_above_cloud_aligned[i]) or np.isnan(price_below_cloud_aligned[i]) or
            np.isnan(tk_bullish_aligned[i]) or np.isnan(tk_bearish_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend filter from Kumo cloud
        trend_12h_uptrend = price_above_cloud_aligned[i]
        trend_12h_downtrend = price_below_cloud_aligned[i]
        
        if position == 0:
            # Long: Bullish TK cross + 12h uptrend + volume spike
            long_signal = tk_bullish_aligned[i] and trend_12h_uptrend and volume_spike[i]
            
            # Short: Bearish TK cross + 12h downtrend + volume spike
            short_signal = tk_bearish_aligned[i] and trend_12h_downtrend and volume_spike[i]
            
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
            # Exit: Bearish TK cross OR price falls below cloud
            if tk_bearish_aligned[i] or not trend_12h_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bullish TK cross OR price rises above cloud
            if tk_bullish_aligned[i] or not trend_12h_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0