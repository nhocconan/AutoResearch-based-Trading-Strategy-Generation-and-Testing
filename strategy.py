#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_1dTrendFilter_v1
Hypothesis: Trade Ichimoku TK cross signals aligned with 1d trend on 6h timeframe.
Only long when Tenkan > Kijun AND price above Kumo cloud AND 1d EMA50 uptrend.
Only short when Tenkan < Kijun AND price below Kumo cloud AND 1d EMA50 downtrend.
Uses Ichimoku components calculated on 6h with 1d trend filter to avoid counter-trend trades.
Target: 15-30 trades/year to minimize fee drag while capturing strong trends.
Works in bull via TK cross + cloud breakout, works in bear via TK cross + cloud breakdown.
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
    
    # Get 1d data for trend regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend regime
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku parameters (standard 9,26,52)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26  # Kumo cloud displacement
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max()
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 displaced 26 periods ahead
    senkou_span_a = ((tenkan + kijun) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 displaced 26 periods ahead
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Align Ichimoku components (no extra displacement needed as align_htf_to_ltf handles completed bar timing)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, prices, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, prices, senkou_span_b.values)
    
    # Determine Kumo cloud boundaries (Senkou Span A/B)
    # For cloud, we need values shifted forward by displacement periods
    # But since we're using current Tenkan/Kijun to calculate Senkou, we align the Senkou directly
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52) and 1d EMA50 (50)
    start_idx = max(52, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend regime
        # Bull regime: price > EMA50
        # Bear regime: price < EMA50
        if close[i] > ema_50_1d_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_50_1d_aligned[i]:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades
        
        # Ichimoku signals
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]  # Tenkan above Kijun
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]  # Tenkan below Kijun
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 0:
            # Long setup: TK bullish AND price above cloud AND bull regime
            long_setup = tk_bullish and price_above_cloud and (regime == 'bull')
            
            # Short setup: TK bearish AND price below cloud AND bear regime
            short_setup = tk_bearish and price_below_cloud and (regime == 'bear')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TK bearish OR price below cloud OR regime turns bearish
            if (tk_bearish or not price_above_cloud) or (regime == 'bear'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TK bullish OR price above cloud OR regime turns bullish
            if (tk_bullish or not price_below_cloud) or (regime == 'bull'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0