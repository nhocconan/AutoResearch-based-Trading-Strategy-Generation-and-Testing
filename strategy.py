#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: 6h Ichimoku cloud breakout filtered by 1w trend (price > 1w SMA50) and volume confirmation (>2x 24-period average).
Uses discrete position sizing (0.25) to limit drawdown in bear markets. Ichimoku provides dynamic support/resistance
via cloud (Senkou Span A/B) and momentum via TK cross. Weekly trend filter ensures we only trade with the higher
timeframe momentum, reducing whipsaws in ranging markets. Volume confirmation adds conviction to breakouts.
Designed for 6h timeframe to capture medium-term trends while avoiding excessive trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # need enough for 1w SMA50 and Ichimoku
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w SMA50 for trend filter ===
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # === Ichimoku components (calculate on 6h data) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Conversion Line (Tenkan-sen): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Base Line (Kijun-sen): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Leading Span A (Senkou Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Leading Span B (Senkou Span B): (52-period high + low)/2 plotted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components (no additional delay needed as they're based on completed candles)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan_sen)  # same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, prices, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, prices, senkou_span_b)
    
    # === Volume filter: 24-period average (6h * 24 = 6 days) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):  # start after Ichimoku components are ready
        # Skip if indicators not ready
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(tenkan_aligned[i]) 
            or np.isnan(kijun_aligned[i]) or np.isnan(span_a_aligned[i]) 
            or np.isnan(span_b_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_average = vol_ma[i]
        
        # Cloud top and bottom
        cloud_top = max(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = min(span_a_aligned[i], span_b_aligned[i])
        
        if position == 0:
            # Volume filter: current volume > 2x 24-period average
            vol_filter = vol_current > 2.0 * vol_average
            
            # Long conditions: price breaks above cloud, TK cross bullish, 1w uptrend, volume
            price_above_cloud = price > cloud_top
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
            weekly_uptrend = price > sma_50_1w_aligned[i]
            
            # Short conditions: price breaks below cloud, TK cross bearish, 1w downtrend, volume
            price_below_cloud = price < cloud_bottom
            tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
            weekly_downtrend = price < sma_50_1w_aligned[i]
            
            # Entry logic
            if price_above_cloud and tk_bullish and weekly_uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif price_below_cloud and tk_bearish and weekly_downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: price closes below cloud or TK cross turns bearish
            if price < cloud_bottom or tenkan_aligned[i] < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above cloud or TK cross turns bullish
            if price > cloud_top or tenkan_aligned[i] > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0