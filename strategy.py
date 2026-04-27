#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Choppiness Index filter (14-period) to avoid whipsaws in ranging markets
    atr_14 = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        atr_14[i] = np.mean(tr[max(0, i-13):i+1]) if i >= 13 else np.nan
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR14) / (n * (max(high) - min(low)))) / log10(n)
    chop = np.full(n, 50.0)  # Default to neutral
    for i in range(14, n):
        if not np.isnan(atr_14[i]):
            atr_sum = np.sum(atr_14[i-13:i+1])
            period_high = np.max(high[i-13:i+1])
            period_low = np.min(low[i-13:i+1])
            if period_high > period_low:
                chop[i] = 100 * np.log10(atr_sum / (14 * (period_high - period_low))) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Ichimoku (52), ATR (14)
    start_idx = max(52, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Ichimoku signals
        tk_cross = tenkan_sen_aligned[i] > kijun_sen_aligned[i]  # Bullish TK cross
        price_above_cloud = (price > senkou_span_a_aligned[i]) and (price > senkou_span_b_aligned[i])
        price_below_cloud = (price < senkou_span_a_aligned[i]) and (price < senkou_span_b_aligned[i])
        
        # Choppiness filter: avoid trading in strong ranging markets (CHOP > 61.8)
        not_choppy = chop[i] <= 61.8
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + not choppy
            if tk_cross and price_above_cloud and not_choppy:
                signals[i] = size
                position = 1
            # Short: TK cross bearish + price below cloud + not choppy
            elif not tk_cross and price_below_cloud and not_choppy:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TK cross bearish OR price below cloud
            if not tk_cross or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK cross bullish OR price above cloud
            if tk_cross or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_Chop"
timeframe = "6h"
leverage = 1.0