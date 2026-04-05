#!/usr/bin/env python3
"""
Experiment #10455: 6h Ichimoku Cloud with 1d Trend Filter
Hypothesis: Ichimoku Tenkan/Kijun cross combined with 1d cloud filter provides
high-probability entries in both bull and bear markets. The 1d cloud acts as
a trend filter (price above/below cloud) while 6h TK cross provides timing.
Works in bull markets (long when price above 1d cloud + TK cross up) and
bear markets (short when price below 1d cloud + TK cross down).
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10455_6h_ichimoku_1d_cloud_filter_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(KUMO_SHIFT)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for cloud filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku for cloud filter
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    tenkan_d, kijun_d, senkou_a_d, senkou_b_d = calculate_ichimoku(daily_high, daily_low, daily_close)
    
    # Cloud boundaries: Senkou Span A and B
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top_d = np.maximum(senkou_a_d, senkou_b_d)
    cloud_bottom_d = np.minimum(senkou_a_d, senkou_b_d)
    
    # Align daily cloud to 6h timeframe
    cloud_top_aligned = align_htf_to_ltf(prices, df_daily, cloud_top_d)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_daily, cloud_bottom_d)
    
    # Calculate 6h Ichimoku for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tenkan_6h, kijun_6h, _, _ = calculate_ichimoku(high, low, close)
    
    # TK cross signals
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_6h <= kijun_6h)  # Crossed up
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_6h >= kijun_6h)  # Crossed down
    
    # Trend filters: price relative to daily cloud
    price_above_cloud = close > cloud_top_aligned
    price_below_cloud = close < cloud_bottom_aligned
    
    # Cloud color (bullish/bearish) - optional filter
    cloud_bullish = senkou_a_d > senkou_b_d
    cloud_bearish = senkou_a_d < senkou_b_d
    cloud_bullish_aligned = align_htf_to_ltf(prices, df_daily, cloud_bullish)
    cloud_bearish_aligned = align_htf_to_ltf(prices, df_daily, cloud_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT) + 1
    
    for i in range(start, n):
        # Skip if daily cloud not available
        if np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (tk_cross_up[i] if not np.isnan(tk_cross_up[i]) else False) and \
                     price_above_cloud[i] and \
                     (cloud_bullish_aligned[i] if not np.isnan(cloud_bullish_aligned[i]) else True)
        
        short_entry = (tk_cross_down[i] if not np.isnan(tk_cross_down[i]) else False) and \
                      price_below_cloud[i] and \
                      (cloud_bearish_aligned[i] if not np.isnan(cloud_bearish_aligned[i]) else True)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit on reverse TK cross or price re-enters cloud
            exit_signal = (tk_cross_down[i] if not np.isnan(tk_cross_down[i]) else False) or \
                          (close[i] < cloud_top_aligned[i] and close[i] > cloud_bottom_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on reverse TK cross or price re-enters cloud
            exit_signal = (tk_cross_up[i] if not np.isnan(tk_cross_up[i]) else False) or \
                          (close[i] < cloud_top_aligned[i] and close[i] > cloud_bottom_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals