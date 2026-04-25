#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm
Hypothesis: 6h Ichimoku cloud twist (Senkou Span A/B cross) with 1d trend filter (price >/<- EMA50) and volume confirmation (>1.5x 20-bar avg). Enters long when Kumo twist bullish (Senkou A crosses above Senkou B) in 1d uptrend, short when bearish twist in 1d downtrend. Uses discrete sizing (0.25) to limit fee churn. Designed for 6h timeframe with ~12-30 trades/year, works in bull/bear by following 1d trend filter.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to avoid look-ahead (Senkou spans are already shifted)
    # We need to align the current values properly
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Kumo twist detection: Senkou A crossing above/below Senkou B
    # Bullish twist: Senkou A crosses above Senkou B (previous A <= previous B and current A > current B)
    # Bearish twist: Senkou A crosses below Senkou B (previous A >= previous B and current A < current B)
    senkou_a_prev = np.roll(senkou_a_aligned, 1)
    senkou_b_prev = np.roll(senkou_b_aligned, 1)
    senkou_a_prev[0] = senkou_a_aligned[0]
    senkou_b_prev[0] = senkou_b_aligned[0]
    
    bullish_twist = (senkou_a_aligned > senkou_b_aligned) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a_aligned < senkou_b_aligned) & (senkou_a_prev >= senkou_b_prev)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 52-period data for Senkou B and 50 for 1d EMA
    start_idx = max(52, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish Kumo twist in 1d uptrend with volume confirmation
            bullish_setup = bullish_twist[i] and (close_1d[i] > ema_50_1d_aligned[i]) and volume_spike[i]
            # Short: bearish Kumo twist in 1d downtrend with volume confirmation
            bearish_setup = bearish_twist[i] and (close_1d[i] < ema_50_1d_aligned[i]) and volume_spike[i]
            
            if bullish_setup:
                signals[i] = 0.25
                position = 1
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: bearish Kumo twist OR trend turns down
            if bearish_twist[i] or (close_1d[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: bullish Kumo twist OR trend turns up
            if bullish_twist[i] or (close_1d[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0