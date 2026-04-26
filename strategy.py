#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend
Hypothesis: Use Ichimoku TK cross on 6h with cloud filter from 1d and weekly trend confirmation. Long when TK crosses above in bullish cloud + 1w uptrend; short when TK crosses below in bearish cloud + 1w downtrend. Uses discrete 0.25 position size. Targets 15-25 trades/year to avoid fee drag.
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
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values +
                  pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values +
                 pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values +
                      pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values) / 2)
    
    # Chikou Span (Lagging Span): close shifted -26 periods (not used for signals to avoid look-ahead)
    
    # Cloud: Senkou Span A and Senkou Span B
    # Bullish cloud: Senkou Span A > Senkou Span B
    # Bearish cloud: Senkou Span A < Senkou Span B
    bullish_cloud = senkou_span_a > senkou_span_b
    bearish_cloud = senkou_span_a < senkou_span_b
    
    # TK Cross signals (using completed candles only)
    # Bullish TK cross: Tenkan-sen crosses above Kijun-sen
    tk_bullish = (tenkan_sen > kijun_sen) & (tenkan_sen <= kijun_sen)  # Using <= for previous bar comparison
    # Bearish TK cross: Tenkan-sen crosses below Kijun-sen
    tk_bearish = (tenkan_sen < kijun_sen) & (tenkan_sen >= kijun_sen)  # Using >= for previous bar comparison
    
    # Fix TK cross logic - proper crossover detection
    tk_bullish = (tenkan_sen > kijun_sen) & (np.roll(tenkan_sen, 1) <= np.roll(kijun_sen, 1))
    tk_bearish = (tenkan_sen < kijun_sen) & (np.roll(tenkan_sen, 1) >= np.roll(kijun_sen, 1))
    # Handle first bar
    tk_bullish[0] = False
    tk_bearish[0] = False
    
    # Multi-timeframe filters
    # 1d HTF for trend context (higher timeframe than 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for reliable trend
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w HTF for major trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA20 for major trend
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Senkou Span B, 50 for 1d EMA, 20 for 1w EMA
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size - reduced to manage drawdown
        
        if position == 0:
            # Flat - look for TK cross with cloud and trend filters
            # Long: Bullish TK cross + bullish cloud (Senkou A > Senkou B) + 1d EMA50 up + 1w EMA20 up
            long_entry = tk_bullish[i] and \
                       bullish_cloud[i] and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[max(i-1, start_idx)]) and \
                       (ema_20_1w_aligned[i] > ema_20_1w_aligned[max(i-1, start_idx)])
            
            # Short: Bearish TK cross + bearish cloud (Senkou A < Senkou B) + 1d EMA50 down + 1w EMA20 down
            short_entry = tk_bearish[i] and \
                        bearish_cloud[i] and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[max(i-1, start_idx)]) and \
                        (ema_20_1w_aligned[i] < ema_20_1w_aligned[max(i-1, start_idx)])
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TK cross turns bearish OR price breaks below cloud
            exit_long = tk_bearish[i] or (close_val < senkou_span_a[i] and close_val < senkou_span_b[i])
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TK cross turns bullish OR price breaks above cloud
            exit_short = tk_bullish[i] or (close_val > senkou_span_a[i] and close_val > senkou_span_b[i])
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend"
timeframe = "6h"
leverage = 1.0