#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_WeeklyTrend_Confirm_v2
Hypothesis: 6h Ichimoku Kumo twist (TK cross) with 1d weekly EMA200 trend filter and volume confirmation.
In bull markets (price > 1d EMA200), take long signals when TK crosses above and price > cloud.
In bear markets (price < 1d EMA200), take short signals when TK crosses below and price < cloud.
Uses discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 6h.
Works in bull via trend-following TK crosses above cloud, in bear via trend-following TK crosses below cloud.
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
    
    # Get 6h data for Ichimoku calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_6h).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_6h).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((pd.Series(high_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                      pd.Series(low_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(displacement)
    
    # Align Ichimoku components to original timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b.values)
    
    # Get 1d data for weekly trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for weekly trend direction
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average (moderate to balance signal quality)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 150
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend regime
        weekly_bull = close[i] > ema_200_1d_aligned[i]
        weekly_bear = close[i] < ema_200_1d_aligned[i]
        
        # Cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK Cross conditions
        tk_cross_above = (tenkan_sen_aligned[i] > kijun_sen_aligned[i]) and (tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
        tk_cross_below = (tenkan_sen_aligned[i] < kijun_sen_aligned[i]) and (tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Regime-based entry logic
            if weekly_bull:
                # Long: TK cross above + price above cloud + volume spike
                long_signal = tk_cross_above and price_above_cloud and vol_spike[i]
            else:  # weekly_bear
                # Short: TK cross below + price below cloud + volume spike
                short_signal = tk_cross_below and price_below_cloud and vol_spike[i]
            
            if weekly_bull and 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif weekly_bear and 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clean up local vars
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: TK cross below OR price drops below cloud bottom
            exit_signal = tk_cross_below or (close[i] < cloud_bottom)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: TK cross above OR price rises above cloud top
            exit_signal = tk_cross_above or (close[i] > cloud_top)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_WeeklyTrend_Confirm_v2"
timeframe = "6h"
leverage = 1.0