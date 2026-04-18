#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Daily Trend Filter and Volume Confirmation
Hypothesis: Ichimoku cloud identifies dynamic support/resistance and trend direction.
Tenkan/Kijun cross provides entry signals, filtered by daily trend and volume spikes.
Designed for 12-37 trades/year on 6h timeframe, works in both bull and bear markets
by only taking trades in the direction of the higher timeframe trend.
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
    
    # Get daily data for Ichimoku (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(df_d['high']).rolling(window=9, min_periods=9).max() + 
                  pd.Series(df_d['low']).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(df_d['high']).rolling(window=26, min_periods=26).max() + 
                 pd.Series(df_d['low']).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((pd.Series(df_d['high']).rolling(window=52, min_periods=52).max() + 
                      pd.Series(df_d['low']).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_d, kijun_sen.values)
    span_a_aligned = align_htf_to_ltf(prices, df_d, senkou_span_a.values)
    span_b_aligned = align_htf_to_ltf(prices, df_d, senkou_span_b.values)
    
    # Daily EMA50 for additional trend filter
    ema_50 = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_d, ema_50)
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(span_a_aligned[i]) or
            np.isnan(span_b_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        span_a = span_a_aligned[i]
        span_b = span_b_aligned[i]
        ema = ema_aligned[i]
        atr_val = atr[i]
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, above daily EMA, volume spike
            if (tenkan > kijun and tenkan <= kijun + 0.0001 and  # Cross occurred recently
                price > cloud_top and 
                price > ema and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud, below daily EMA, volume spike
            elif (tenkan < kijun and tenkan >= kijun - 0.0001 and  # Cross occurred recently
                  price < cloud_bottom and 
                  price < ema and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: Tenkan crosses below Kijun or price returns to cloud bottom
            if tenkan < kijun or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: Tenkan crosses above Kijun or price returns to cloud top
            if tenkan > kijun or price > cloud_top:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0