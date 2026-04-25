#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_Breakout
Hypothesis: On 6h timeframe, Ichimoku cloud acts as dynamic support/resistance with TK cross as momentum trigger. Price breaking above/below cloud with TK cross alignment captures strong trends while cloud filter prevents whipsaws in sideways markets. Uses 12h HTF for cloud calculation to ensure completed bars. Works in bull markets via long breaks above cloud and bear markets via short breaks below cloud. Discrete sizing (0.25) minimizes fee churn. Target 50-150 trades over 4 years.
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
    
    # Get 12h data for Ichimoku cloud calculation (primary HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_tenkan = pd.Series(high_12h).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    low_tenkan = pd.Series(low_12h).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_kijun = pd.Series(high_12h).rolling(window=kijun_period, min_periods=kijun_period).max().values
    low_kijun = pd.Series(low_12h).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_senkou_b = pd.Series(high_12h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    low_senkou_b = pd.Series(low_12h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe (with displacement for forward shift)
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a, additional_delay_bars=displacement)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b, additional_delay_bars=displacement)
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(senkou_span_b_period + displacement, 20)  # Ichimoku + vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Cloud boundaries (Senkou Span A and B form the cloud)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK cross: Tenkan-sen crossing above/below Kijun-sen
        tk_cross_up = tenkan_val > kijun_val
        tk_cross_down = tenkan_val < kijun_val
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Price breaks cloud with TK cross and volume
            # Long: price breaks above cloud with bullish TK cross and volume
            long_signal = (close_val > cloud_top) and tk_cross_up and volume_confirm
            # Short: price breaks below cloud with bearish TK cross and volume
            short_signal = (close_val < cloud_bottom) and tk_cross_down and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price re-enters cloud (close below cloud top)
            if close_val < cloud_top:
                signals[i] = 0.0
                position = 0
            # 2. Bearish TK cross (tenkan < kijun)
            elif tenkan_val < kijun_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price re-enters cloud (close above cloud bottom)
            if close_val > cloud_bottom:
                signals[i] = 0.0
                position = 0
            # 2. Bullish TK cross (tenkan > kijun)
            elif tenkan_val > kijun_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_Breakout"
timeframe = "6h"
leverage = 1.0