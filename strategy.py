#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Daily Trend Filter + Volume Confirmation
Hypothesis: Ichimoku cloud provides dynamic support/resistance and trend direction.
In bull markets, price stays above cloud; in bear markets, price stays below cloud.
TK (Tenkan/Kijun) cross provides timely entry signals aligned with higher timeframe trend.
Daily trend filter ensures we only trade in direction of higher timeframe momentum.
Volume confirmation filters out weak breakouts. Designed for 6h timeframe to capture
swing trades with lower frequency (~20-50 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter: EMA(50)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Shift will be handled by alignment function
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = (pd.Series(high).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low).rolling(window=52, min_periods=52).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back (not needed for signals)
    
    # Align Ichimoku components to lower timeframe (with proper shift for forward-looking components)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after Ichimoku warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR TK cross turns bearish OR trend reverses
            if (close[i] < cloud_bottom or 
                tenkan_sen_aligned[i] < kijun_sen_aligned[i] or
                close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR TK cross turns bullish OR trend reverses
            if (close[i] > cloud_top or 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] or
                close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above cloud, TK bullish cross, aligned with daily trend
            if (close[i] > cloud_top and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and  # TK cross bullish
                close[i] > ema_50_1d_aligned[i] and  # Above daily EMA (uptrend)
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price below cloud, TK bearish cross, aligned with daily trend
            elif (close[i] < cloud_bottom and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and  # TK cross bearish
                  close[i] < ema_50_1d_aligned[i] and  # Below daily EMA (downtrend)
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals