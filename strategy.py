#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm_v2
Hypothesis: 6h Ichimoku TK cross with 1w cloud filter and volume confirmation.
Long when TK crosses above cloud in bullish weekly trend (price > weekly Senkou Span B).
Short when TK crosses below cloud in bearish weekly trend (price < weekly Senkou Span B).
Volume confirmation (1.5x average) reduces false signals. Discrete sizing (0.25) limits fee drag.
Designed to work in both bull and bear markets by aligning with weekly trend direction.
Timeframe: 6h, uses 1w HTF for trend filter and cloud calculation.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for Ichimoku cloud and trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w OHLC for Ichimoku calculation ===
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(df_1w_high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1w_low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(df_1w_high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1w_low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(df_1w_high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1w_low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods behind
    # Not used for signals as it requires future data
    
    # Align 1w Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # === 6h OHLC for price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) 
            or np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        vol_avg = vol_ma[i]
        
        # Identify cloud boundaries (top and bottom of cloud)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # TK cross: Tenkan-sen crossing Kijun-sen
        # Bullish TK cross: Tenkan crosses above Kijun
        # Bearish TK cross: Tenkan crosses below Kijun
        bullish_tk_cross = tenkan > kijun
        bearish_tk_cross = tenkan < kijun
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Long conditions: bullish TK cross, price above cloud, bullish weekly trend (price > Senkou Span B)
            long_condition = (bullish_tk_cross and 
                            price > cloud_top and 
                            price > senkou_span_b_aligned[i] and 
                            volume_confirmed)
            
            # Short conditions: bearish TK cross, price below cloud, bearish weekly trend (price < Senkou Span B)
            short_condition = (bearish_tk_cross and 
                             price < cloud_bottom and 
                             price < senkou_span_b_aligned[i] and 
                             volume_confirmed)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: bearish TK cross or price drops below cloud bottom
            if bearish_tk_cross or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish TK cross or price rises above cloud top
            if bullish_tk_cross or price > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0