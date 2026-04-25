#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with Weekly Kumo Twist and Volume Confirmation
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. Weekly Kumo twist (Senkou Span A/B cross) indicates major trend change. Breakouts in direction of weekly trend with volume confirmation capture strong momentum while avoiding false signals. Works in bull/bear via trend filter. Target: 12-30 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past tenkan periods
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past kijun periods
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted senkou periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(senkou)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past senkou*2 periods shifted senkou ahead
    senkou_span_b = ((pd.Series(high).rolling(window=senkou*2, min_periods=senkou*2).max() + 
                      pd.Series(low).rolling(window=senkou*2, min_periods=senkou*2).min()) / 2).shift(senkou)
    
    # Chikou Span (Lagging Span): Close shifted senkou periods behind
    chikou_span = pd.Series(close).shift(-senkou)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 52 or len(df_1w) < 52:
        return np.zeros(n)
    
    # 1d Ichimoku
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # 1w Ichimoku for Kumo twist (Senkou A/B cross)
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values
    )
    
    # Align 1d Ichimoku to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Align 1w Ichimoku to 6h timeframe (Kumo twist detection)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Kumo twist: Senkou Span A crosses above/below Senkou Span B on weekly
    kumo_twist_bullish = senkou_a_1w_aligned > senkou_b_1w_aligned  # Bullish twist: A above B
    kumo_twist_bearish = senkou_a_1w_aligned < senkou_b_1w_aligned  # Bearish twist: A below B
    
    # 6h Donchian(20) for breakout signals
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(52, donchian_window, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(kumo_twist_bullish[i]) or 
            np.isnan(kumo_twist_bearish[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Ichimoku signals
        price_above_cloud = (curr_close > senkou_a_1d_aligned[i]) and (curr_close > senkou_b_1d_aligned[i])
        price_below_cloud = (curr_close < senkou_a_1d_aligned[i]) and (curr_close < senkou_b_1d_aligned[i])
        tenkan_cross_above_kijun = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tenkan_cross_below_kijun = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Kumo twist alignment + Ichimoku signal + volume
            # Long: bullish Kumo twist AND price above cloud AND Tenkan > Kijun AND volume spike
            long_entry = (kumo_twist_bullish[i] and price_above_cloud and 
                         tenkan_cross_above_kijun and vol_spike)
            # Short: bearish Kumo twist AND price below cloud AND Tenkan < Kijun AND volume spike
            short_entry = (kumo_twist_bearish[i] and price_below_cloud and 
                          tenkan_cross_below_kijun and vol_spike)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below cloud OR Tenkan crosses below Kijun (trend weakening)
            if (price_below_cloud or tenkan_cross_below_kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above cloud OR Tenkan crosses above Kijun (trend weakening)
            if (price_above_cloud or tenkan_cross_above_kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyKumoTwist_VolumeSpike"
timeframe = "6h"
leverage = 1.0