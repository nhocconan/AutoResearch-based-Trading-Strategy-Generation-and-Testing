#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with 1d Cloud Filter and Volume Confirmation v1
Hypothesis: Ichimoku provides robust trend/direction signals; combining TK cross
with cloud position from 1d timeframe filters counter-trend trades. Volume
confirms breakout strength. Designed for 50-150 trades over 4 years to minimize
fee drag while adapting to bull/bear markets via cloud position filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_kumo_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for cloud filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    highest_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    highest_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    highest_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Align to 6h timeframe (shift by 1 for completed candles)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TK Cross signals (6h)
    tk_cross_up = np.zeros(n, dtype=bool)
    tk_cross_down = np.zeros(n, dtype=bool)
    tenkan_6h = pd.Series(close).rolling(window=9, min_periods=9).apply(
        lambda x: (x[-1] + pd.Series(high).rolling(9, min_periods=9).apply(
            lambda y: y[-1] if len(y)==9 else np.nan, raw=True).iloc[-1] 
            + pd.Series(low).rolling(9, min_periods=9).apply(
                lambda z: z[-1] if len(z)==9 else np.nan, raw=True).iloc[-1])/2 if len(x)==9 else np.nan, 
        raw=True).values
    kijun_6h = pd.Series(close).rolling(window=26, min_periods=26).apply(
        lambda x: (x[-1] + pd.Series(high).rolling(26, min_periods=26).apply(
            lambda y: y[-1] if len(y)==26 else np.nan, raw=True).iloc[-1] 
        + pd.Series(low).rolling(26, min_periods=26).apply(
            lambda z: z[-1] if len(z)==26 else np.nan, raw=True).iloc[-1])/2 if len(x)==26 else np.nan, 
        raw=True).values
    
    # Simplified TK cross calculation
    tenkan_simplified = (pd.Series(high).rolling(9, min_periods=9).max() + 
                         pd.Series(low).rolling(9, min_periods=9).min()) / 2
    kijun_simplified = (pd.Series(high).rolling(26, min_periods=26).max() + 
                        pd.Series(low).rolling(26, min_periods=26).min()) / 2
    tk_cross_up = (tenkan_simplified > kijun_simplified) & (tenkan_simplified.shift(1) <= kijun_simplified.shift(1))
    tk_cross_down = (tenkan_simplified < kijun_simplified) & (tenkan_simplified.shift(1) >= kijun_simplified.shift(1))
    
    # Cloud breakout signals
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(52, 26, 20)  # For Ichimoku and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_simplified.iloc[i]) or np.isnan(kijun_simplified.iloc[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma[i]) or np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite TK cross or cloud reversal
        if position == 1:  # long position
            # Exit: TK cross down OR price closes below cloud
            if tk_cross_down.iloc[i] or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross up OR price closes above cloud
            if tk_cross_up.iloc[i] or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + cloud position + volume
            bull_entry = tk_cross_up.iloc[i] and close[i] > cloud_top[i] and volume[i] > vol_ma[i] * 1.5
            bear_entry = tk_cross_down.iloc[i] and close[i] < cloud_bottom[i] and volume[i] > vol_ma[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals