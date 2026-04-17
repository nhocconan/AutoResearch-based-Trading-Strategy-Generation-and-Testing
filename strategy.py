#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d EMA50 trend filter and volume confirmation.
Long when price above Ichimoku cloud, TK cross bullish, price > 1d EMA50, and volume > 1.5x average.
Short when price below Ichimoku cloud, TK cross bearish, price < 1d EMA50, and volume > 1.5x average.
Exit when price crosses back into cloud or volume drops below average.
Ichimoku provides dynamic support/resistance, TK cross signals momentum, EMA50 filters trend direction.
Volume confirmation reduces fakeouts. Designed to capture trends in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
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
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_6h).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    low_9 = pd.Series(low_6h).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan_sen = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max()
    low_26 = pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun_sen = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).values
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    high_52 = pd.Series(high_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    low_52 = pd.Series(low_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b = ((high_52 + low_52) / 2).values
    
    # Get 1d data for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume average (20-period) on 6h
    volume_6h = df_6h['volume'].values
    volume_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for Ichimoku (52+26+buffer)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        tk = tenkan_sen_aligned[i]
        kj = kijun_sen_aligned[i]
        sa = senkou_span_a_aligned[i]
        sb = senkou_span_b_aligned[i]
        ema50 = ema_50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Ichimoku Cloud boundaries
        upper_cloud = max(sa, sb)
        lower_cloud = min(sa, sb)
        
        # TK Cross
        tk_cross_bullish = tk > kj
        tk_cross_bearish = tk < kj
        
        if position == 0:
            # Long: price above cloud, TK bullish, price > EMA50, volume > 1.5x avg
            if (price > upper_cloud and 
                tk_cross_bullish and 
                price > ema50 and 
                vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, TK bearish, price < EMA50, volume > 1.5x avg
            elif (price < lower_cloud and 
                  tk_cross_bearish and 
                  price < ema50 and 
                  vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses into cloud OR volume drops below average
            if price < upper_cloud or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses into cloud OR volume drops below average
            if price > lower_cloud or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuCloud_Volume_EMA50_Filter"
timeframe = "6h"
leverage = 1.0