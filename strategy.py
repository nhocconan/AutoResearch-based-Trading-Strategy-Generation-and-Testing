#!/usr/bin/env python3
name = "6h_Ichimoku_TF_Align_Trend"
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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    kumo_shift = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_span_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Future cloud boundaries (shifted forward by kumo_shift periods)
    senkou_a_leading = np.roll(senkou_span_a, kumo_shift)
    senkou_b_leading = np.roll(senkou_span_b, kumo_shift)
    # First kumo_shift values are invalid due to roll
    senkou_a_leading[:kumo_shift] = np.nan
    senkou_b_leading[:kumo_shift] = np.nan
    
    # Current cloud (for price position) - use unshifted Senkou spans
    senkou_a_current = senkou_span_a
    senkou_b_current = senkou_span_b
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, kumo_shift) + 10
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a_leading[i]) or np.isnan(senkou_b_leading[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine if price is above or below cloud
        cloud_top = np.maximum(senkou_a_leading[i], senkou_b_leading[i])
        cloud_bottom = np.minimum(senkou_a_leading[i], senkou_b_leading[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross
        tk_cross_bullish = tenkan[i] > kijun[i]
        tk_cross_bearish = tenkan[i] < kijun[i]
        
        if position == 0:
            # Long: Price above cloud + TK bullish cross + 1d uptrend + volume
            if price_above_cloud and tk_cross_bullish and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + TK bearish cross + 1d downtrend + volume
            elif price_below_cloud and tk_cross_bearish and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below cloud OR TK bearish cross
            if price_below_cloud or not tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above cloud OR TK bullish cross
            if price_above_cloud or tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals