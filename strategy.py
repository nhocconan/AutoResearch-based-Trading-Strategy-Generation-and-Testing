#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_12hTrend_v1
Hypothesis: On 6h timeframe, trade long when price breaks above Ichimoku cloud (Senkou Span A/B) and short when breaks below cloud, 
filtered by 12h EMA50 trend. Ichimoku cloud acts as dynamic support/resistance that adapts to volatility. 
The 12h EMA50 trend filter ensures alignment with higher-timeframe direction, improving performance in both bull and bear markets. 
ATR-based stoploss (2*ATR) manages risk. Targeting 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Ichimoku cloud calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # Need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_12h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_12h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_12h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_12h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_52 + min_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (completed 12h bar only)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for stoploss calculation (6h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Ichimoku (52), EMA50 (50), ATR (14)
    start_idx = max(52, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        senkou_a_val = senkou_span_a_aligned[i]
        senkou_b_val = senkou_span_b_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long: price breaks above cloud top, above 1d EMA50
            long_signal = (close_val > cloud_top) and (close_val > ema_50_val)
            
            # Short: price breaks below cloud bottom, below 1d EMA50
            short_signal = (close_val < cloud_bottom) and (close_val < ema_50_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below cloud bottom OR ATR stoploss (2*ATR below entry)
            if (close_val < cloud_bottom) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above cloud top OR ATR stoploss (2*ATR above entry)
            if (close_val > cloud_top) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_12hTrend_v1"
timeframe = "6h"
leverage = 1.0