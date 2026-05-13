#!/usr/bin/env python3
# 6h_Ichimoku_Multi_Trend_Filter
# Hypothesis: Ichimoku cloud twist (Tenkan/Kijun cross) with Senkou Span color change + price outside cloud confirmation. Uses daily timeframe for cloud color to avoid look-ahead. Trades only when price is above/below cloud AND in correct trend regime. Designed for low trade frequency (<30/year) to minimize fee drag while capturing major trends in both bull and bear markets.

name = "6h_Ichimoku_Multi_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku components (Tenkan, Kijun, Senkou Span)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_high = df_1d['high'].rolling(window=9, min_periods=9).max().values
    tenkan_low = df_1d['low'].rolling(window=9, min_periods=9).min().values
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_high = df_1d['high'].rolling(window=26, min_periods=26).max().values
    kijun_low = df_1d['low'].rolling(window=26, min_periods=26).min().values
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_high = df_1d['high'].rolling(window=52, min_periods=52).max().values
    senkou_low = df_1d['low'].rolling(window=52, min_periods=52).min().values
    senkou_span_b = (senkou_high + senkou_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Daily trend filter: price > 200 EMA for uptrend, price < 200 EMA for downtrend
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if Ichimoku data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate cloud boundaries
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_green = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]  # bullish cloud
        
        if position == 0:
            # LONG: Tenkan crosses above Kijun, price above cloud, cloud bullish, price above 200 EMA, volume confirmed
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and  # cross just happened
                close[i] > cloud_top and
                cloud_green and
                close[i] > ema_200_1d_aligned[i] and
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun, price below cloud, cloud red, price below 200 EMA, volume confirmed
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and  # cross just happened
                  close[i] < cloud_bottom and
                  not cloud_green and
                  close[i] < ema_200_1d_aligned[i] and
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun OR price falls below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]) or \
               close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun OR price rises above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]) or \
               close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals