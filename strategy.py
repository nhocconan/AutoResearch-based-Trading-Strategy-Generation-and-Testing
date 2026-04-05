#!/usr/bin/env python3
"""
Experiment #9987: 6h Ichimoku Cloud with Daily Trend Filter
Hypothesis: Ichimoku Kijun/Tenkan cross combined with price position relative to daily cloud
provides high-probability trend continuation trades. Works in bull markets (price above cloud,
bullish TK cross) and bear markets (price below cloud, bearish TK cross). The daily cloud acts
as a trend filter to avoid counter-trend trades. Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9987_6h_ichimoku_daily_trend"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9      # Tenkan-sen (Conversion Line)
KIJUN_PERIOD = 26      # Kijun-sen (Base Line)
SENKOU_SPAN_B_PERIOD = 52  # Senkou Span B
TK_CROSS_CONFIRM = 3   # Bars to confirm TK cross
SIGNAL_SIZE = 0.25

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over TENKAN_PERIOD
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() +
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over KIJUN_PERIOD
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() +
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward KIJUN_PERIOD
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over SENKOU_SPAN_B_PERIOD shifted forward KIJUN_PERIOD
    senkou_span_b = ((pd.Series(high).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max() +
                      pd.Series(low).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted back KIJUN_PERIOD
    chikou_span = pd.Series(close)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def calculate_daily_cloud(high_daily, low_daily):
    """Calculate daily Ichimoku cloud (Senkou Span A and B)"""
    # Daily Tenkan-sen
    daily_tenkan = (pd.Series(high_daily).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() +
                    pd.Series(low_daily).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Daily Kijun-sen
    daily_kijun = (pd.Series(high_daily).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() +
                   pd.Series(low_daily).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Daily Senkou Span A
    daily_senkou_a = ((daily_tenkan + daily_kijun) / 2)
    
    # Daily Senkou Span B
    daily_senkou_b = ((pd.Series(high_daily).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max() +
                       pd.Series(low_daily).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()) / 2)
    
    return daily_senkou_a.values, daily_senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku cloud for trend filter
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_senkou_a, daily_senkou_b = calculate_daily_cloud(daily_high, daily_low)
    
    # Align daily cloud to 6h timeframe
    daily_senkou_a_aligned = align_htf_to_ltf(prices, df_daily, daily_senkou_a)
    daily_senkou_b_aligned = align_htf_to_ltf(prices, df_daily, daily_senkou_b)
    
    # Calculate 6h Ichimoku
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = calculate_ichimoku(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    tk_cross_bullish_count = 0
    tk_cross_bearish_count = 0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_SPAN_B_PERIOD) + KIJUN_PERIOD
    
    for i in range(start, n):
        # Skip if daily cloud not available
        if np.isnan(daily_senkou_a_aligned[i]) or np.isnan(daily_senkou_b_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Determine cloud top and bottom (Senkou Span A and B)
        cloud_top = max(daily_senkou_a_aligned[i], daily_senkou_b_aligned[i])
        cloud_bottom = min(daily_senkou_a_aligned[i], daily_senkou_b_aligned[i])
        
        # Price relative to daily cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross on 6h chart
        tk_cross_bullish = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_bearish = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        # Count consecutive bars of TK cross for confirmation
        if tk_cross_bullish:
            tk_cross_bullish_count += 1
            tk_cross_bearish_count = 0
        elif tk_cross_bearish:
            tk_cross_bearish_count += 1
            tk_cross_bullish_count = 0
        else:
            tk_cross_bullish_count = 0
            tk_cross_bearish_count = 0
        
        tk_cross_bullish_confirmed = tk_cross_bullish_count >= TK_CROSS_CONFIRM
        tk_cross_bearish_confirmed = tk_cross_bearish_count >= TK_CROSS_CONFIRM
        
        # Entry conditions
        long_entry = price_above_cloud and tk_cross_bullish_confirmed
        short_entry = price_below_cloud and tk_cross_bearish_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit if price closes below cloud or bearish TK cross confirmed
            if not price_above_cloud or tk_cross_bearish_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit if price closes above cloud or bullish TK cross confirmed
            if not price_below_cloud or tk_cross_bullish_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals