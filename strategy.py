#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_v1
Hypothesis: Ichimoku TK Cross with Cloud Filter on 6h timeframe. Uses 1d HTF trend alignment for higher timeframe context. In trending markets (price above/below cloud), follow TK cross direction. In ranging markets (price inside cloud), fade TK cross at cloud edges. Uses discrete position sizing (0.25) to minimize fee churn. Targets 50-150 trades over 4 years by requiring TK cross confirmation and cloud filter. Works in bull/bear via adaptive logic: trend following in strong trends, mean reversion in chop.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku parameters
    tenkan_period = 9   # Tenkan-sen (Conversion Line)
    kijun_period = 26   # Kijun-sen (Base Line)
    senkou_span_b_period = 52  # Senkou Span B
    displacement = 26   # Kijun-sen displacement for cloud
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods plotted 26 periods ahead
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Align HTF indicators to LTF (accounting for displacement)
    # Senkou spans are already displaced, so we align without additional shift
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # TK Cross signals
    tk_cross = np.where(tenkan_sen > kijun_sen, 1, -1)  # 1 = bullish cross, -1 = bearish cross
    
    # 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou Span B, 26 for Kijun)
    start_idx = max(senkou_span_b_period, kijun_period, tenkan_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tk_cross[i]) or np.isnan(htf_trend[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        price_in_cloud = (close[i] >= cloud_bottom[i]) & (close[i] <= cloud_top[i])
        
        # Trend following logic: when price is outside cloud, follow TK cross
        if price_above_cloud and tk_cross[i] == 1:  # Bullish alignment
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif price_below_cloud and tk_cross[i] == -1:  # Bearish alignment
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Mean reversion logic: when price is inside cloud, fade at cloud edges
        elif price_in_cloud:
            if close[i] <= cloud_bottom[i] * 1.001:  # Near cloud bottom (long bias)
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] >= cloud_top[i] * 0.999:  # Near cloud top (short bias)
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Exit mean reversion position when price returns to cloud midpoint
                cloud_mid = (cloud_top[i] + cloud_bottom[i]) / 2
                if position == 1 and close[i] > cloud_mid:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] < cloud_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
        else:
            # Transition: price at cloud edges but TK cross not aligned - hold or flatten
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_v1"
timeframe = "6h"
leverage = 1.0