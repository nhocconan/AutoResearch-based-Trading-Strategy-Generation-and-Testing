#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Weekly Pivot Direction + Volume Spike
Hypothesis: Ichimoku TK cross signals momentum direction, weekly pivot provides institutional bias,
volume spike confirms participation. Long when price above cloud, TK bullish, above weekly pivot with volume.
Short when price below cloud, TK bearish, below weekly pivot with volume. Works in bull/bear via cloud filter.
6h timeframe targets 12-37 trades/year (50-150 over 4 years).
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
    
    # Load 1d data ONCE before loop for Ichimoku and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Ichimoku
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(df_1d['high']).rolling(window=9, min_periods=9).mean() +
                  pd.Series(df_1d['low']).rolling(window=9, min_periods=9).mean()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(df_1d['high']).rolling(window=26, min_periods=26).mean() +
                 pd.Series(df_1d['low']).rolling(window=26, min_periods=26).mean()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(df_1d['high']).rolling(window=52, min_periods=52).mean() +
                      pd.Series(df_1d['low']).rolling(window=52, min_periods=52).mean()) / 2)
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe (wait for completed 1d bar + 26-bar shift for cloud)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b, additional_delay_bars=26)
    
    # Weekly pivot levels from previous week (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly pivot calculation
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Weekly pivot: P = (H + L + C)/3
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    # Weekly R1 = (2*P) - L, S1 = (2*P) - H
    weekly_r1 = (2 * weekly_pivot) - prev_week_low
    weekly_s1 = (2 * weekly_pivot) - prev_week_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(52, 20)  # Ichimoku 52-period, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Ichimoku signals
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        # Cloud: Senkou Span A and B form the cloud
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        if position == 0:
            # Look for entry signals
            # Long: price above cloud, TK bullish, above weekly pivot, volume spike
            long_entry = (price_above_cloud and tk_bullish and 
                         (curr_close > weekly_pivot_aligned[i]) and vol_spike)
            # Short: price below cloud, TK bearish, below weekly pivot, volume spike
            short_entry = (price_below_cloud and tk_bearish and 
                          (curr_close < weekly_pivot_aligned[i]) and vol_spike)
            
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
            # Exit: price falls below cloud OR TK turns bearish
            if (price_below_cloud or not tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above cloud OR TK turns bullish
            if (price_above_cloud or tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0