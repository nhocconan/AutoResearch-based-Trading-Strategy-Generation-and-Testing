#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Filter
Hypothesis: On 6h timeframe, use Ichimoku cloud twist (Senkou Span A/B cross) from 1d as regime filter.
Enter long when price breaks above Kumo cloud with bullish twist (Senkou A > Senkou B), short when price breaks below Kumo with bearish twist (Senkou A < Senkou B).
Add volume confirmation to avoid false breakouts. Uses discrete sizing (0.25) to minimize fees.
Target: 20-40 trades/year. Works in bull via breakouts with cloud support, in bear via breakdowns with cloud resistance.
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
    
    # Get 1d data for Ichimoku calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters: 9, 26, 52
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current Kumo (cloud) boundaries: Senkou A and B shifted back 26 periods
    # So we need to shift Senkou A and B BACK by 26 to get today's cloud
    senkou_a_today = np.concatenate([np.full(26, np.nan), senkou_a[:-26]]) if len(senkou_a) > 26 else np.full_like(senkou_a, np.nan)
    senkou_b_today = np.concatenate([np.full(26, np.nan), senkou_b[:-26]]) if len(senkou_b) > 26 else np.full_like(senkou_b, np.nan)
    
    # Kumo twist: Senkou A > Senkou B = bullish, Senkou A < Senkou B = bearish
    kumo_twist_bullish = senkou_a_today > senkou_b_today
    kumo_twist_bearish = senkou_a_today < senkou_b_today
    
    # Align Ichimoku components to 6h timeframe
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_today)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_today)
    kumo_top = np.maximum(senkou_a_aligned, senkou_b_aligned)  # upper cloud boundary
    kumo_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)  # lower cloud boundary
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bearish.astype(float))
    
    # Volume confirmation: today's volume vs 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Kumo top, bullish twist, volume above average
            long_signal = (close[i] > kumo_top[i]) and (kumo_twist_bullish_aligned[i] > 0.5) and (volume[i] > vol_ma_20[i])
            # Short: price breaks below Kumo bottom, bearish twist, volume above average
            short_signal = (close[i] < kumo_bottom[i]) and (kumo_twist_bearish_aligned[i] > 0.5) and (volume[i] > vol_ma_20[i])
            
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
            # Exit when price closes below Kumo bottom (cloud break) or twist turns bearish
            exit_signal = (close[i] < kumo_bottom[i]) or (kumo_twist_bearish_aligned[i] > 0.5)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price closes above Kumo top (cloud break) or twist turns bullish
            exit_signal = (close[i] > kumo_top[i]) or (kumo_twist_bullish_aligned[i] > 0.5)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Filter"
timeframe = "6h"
leverage = 1.0