#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTK_Cross_v2
Hypothesis: Ichimoku cloud (from 1d) acts as dynamic support/resistance. TK cross (Tenkan/Kijun) on 6h provides entry signals in direction of cloud (bullish if price above cloud, bearish if below). Weekly trend filter (price vs weekly EMA50) avoids counter-trend trades. Designed for low frequency (~15-30 trades/year) to work in both bull and bear regimes by requiring alignment with higher timeframe structure and momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 52 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === Ichimoku cloud from daily data ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (no additional delay needed for cloud)
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === Weekly trend filter: EMA50 on weekly data ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        tenkan = tenkan_sen_6h[i]
        kijun = kijun_sen_6h[i]
        span_a = senkou_a_6h[i]
        span_b = senkou_b_6h[i]
        weekly_ema = ema_50_1w_aligned[i]
        
        # Cloud boundaries: top is max(span_a, span_b), bottom is min(span_a, span_b)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_bullish_cross = tenkan > kijun and tenkan_6h_prev <= kijun_6h_prev if i > 100 else False
            # Price above cloud + bullish TK cross + price above weekly EMA (bullish bias)
            if price_close > cloud_top and price_close > weekly_ema:
                # Check for TK cross using previous values
                if i > 100:
                    tenkan_prev = tenkan_sen_6h[i-1]
                    kijun_prev = kijun_sen_6h[i-1]
                    if tenkan > kijun and tenkan_prev <= kijun_prev:
                        signals[i] = 0.25
                        position = 1
            # Bearish TK cross: Tenkan crosses below Kijun
            elif price_close < cloud_bottom and price_close < weekly_ema:
                if i > 100:
                    tenkan_prev = tenkan_sen_6h[i-1]
                    kijun_prev = kijun_sen_6h[i-1]
                    if tenkan < kijun and tenkan_prev >= kijun_prev:
                        signals[i] = -0.25
                        position = -1
        
        elif position != 0:
            # Exit when price re-enters cloud or TK cross reverses
            if position == 1:
                # Exit long: price falls below cloud OR bearish TK cross
                if price_close < cloud_top:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price rises above cloud OR bullish TK cross
                if price_close > cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        
        # Store previous values for TK cross detection
        if i == 100:
            tenkan_6h_prev = tenkan_sen_6h[i]
            kijun_6h_prev = kijun_sen_6h[i]
        elif i > 100:
            tenkan_6h_prev = tenkan_sen_6h[i-1]
            kijun_6h_prev = kijun_sen_6h[i-1]
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTK_Cross_v2"
timeframe = "6h"
leverage = 1.0