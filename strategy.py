#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_12hTrendFilter_v1
Hypothesis: Ichimoku Kumo twist (Senkou Span A/B cross) indicates regime shift. Combined with 12h EMA50 trend filter to avoid counter-trend trades. Works in bull/bear by only taking trades in direction of higher timeframe trend. Low frequency due to twist rarity + trend alignment requirement.
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
    volume = prices['volume'].values
    
    # Ichimoku parameters (9, 26, 52)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    kumo_shift = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    senkou_span_a = np.roll(senkou_span_a, -kumo_shift)  # shift ahead
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = (pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    senkou_span_b = np.roll(senkou_span_b, -kumo_shift)  # shift ahead
    
    # Current Kumo (cloud) values: Senkou Span A/B from 26 periods ago
    senkou_span_a_now = np.roll(senkou_span_a, kumo_shift)  # shift back to current
    senkou_span_b_now = np.roll(senkou_span_b, kumo_shift)  # shift back to current
    
    # Kumo Twist detection: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou Span A crosses above Senkou Span B
    # Bearish twist: Senkou Span A crosses below Senkou Span B
    senkou_span_a_prev = np.roll(senkou_span_a_now, 1)
    senkou_span_b_prev = np.roll(senkou_span_b_now, 1)
    senkou_span_a_prev[0] = np.nan
    senkou_span_b_prev[0] = np.nan
    
    bullish_twist = (senkou_span_a_now > senkou_span_b_now) & (senkou_span_a_prev <= senkou_span_b_prev)
    bearish_twist = (senkou_span_a_now < senkou_span_b_now) & (senkou_span_a_prev >= senkou_span_b_prev)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52+26), 12h EMA (50), volume MA (20)
    start_idx = max(senkou_span_b_period + kumo_shift, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_span_a_now[i]) or 
            np.isnan(senkou_span_b_now[i]) or 
            np.isnan(bullish_twist[i]) or 
            np.isnan(bearish_twist[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: bullish Kumo twist, price above cloud, uptrend (close > 12h EMA50), volume confirmation
            price_above_cloud = close_val > max(senkou_span_a_now[i], senkou_span_b_now[i])
            long_signal = bullish_twist[i] and price_above_cloud and (close_val > ema_50_12h_val) and (volume_val > 1.5 * vol_ma_val)
            
            # Short: bearish Kumo twist, price below cloud, downtrend (close < 12h EMA50), volume confirmation
            price_below_cloud = close_val < min(senkou_span_a_now[i], senkou_span_b_now[i])
            short_signal = bearish_twist[i] and price_below_cloud and (close_val < ema_50_12h_val) and (volume_val > 1.5 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: bearish Kumo twist or price closes below cloud or trend reversal
            price_below_cloud = close_val < min(senkou_span_a_now[i], senkou_span_b_now[i])
            if bearish_twist[i] or price_below_cloud or (close_val < ema_50_12h_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: bullish Kumo twist or price closes above cloud or trend reversal
            price_above_cloud = close_val > max(senkou_span_a_now[i], senkou_span_b_now[i])
            if bullish_twist[i] or price_above_cloud or (close_val > ema_50_12h_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_12hTrendFilter_v1"
timeframe = "6h"
leverage = 1.0