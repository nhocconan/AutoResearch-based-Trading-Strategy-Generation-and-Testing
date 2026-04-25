#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Breakout_1dTrend_VolumeConfirm
Hypothesis: Ichimoku cloud breakout with 1d trend filter and volume confirmation on 6h timeframe.
- Uses Kumo (cloud) twist from 1d as trend filter (bullish when Senkou Span A > Senkou Span B)
- Entry: price breaks above/below Kumo + volume spike + Kumo twist alignment
- Exit: price re-enters Kumo or trend reversal
- Designed for 12-37 trades/year (50-150 over 4 years) on 6h timeframe
- Works in bull markets via cloud breakouts and bear markets via trend alignment with cloud filter
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
    
    # 1d data for Ichimoku calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = df_1d['high'].rolling(window=9, min_periods=9).max().values
    period9_low = df_1d['low'].rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = df_1d['high'].rolling(window=26, min_periods=26).max().values
    period26_low = df_1d['low'].rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = df_1d['high'].rolling(window=52, min_periods=52).max().values
    period52_low = df_1d['low'].rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo twist: bullish when Senkou Span A > Senkou Span B
    kumo_twist_bullish = senkou_span_a > senkou_span_b
    
    # Align Ichimoku components to 6h timeframe (completed 1d bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    kumo_twist_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish.astype(float))
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations (52 for Senkou Span B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(kumo_twist_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Kumo boundaries (upper and lower cloud)
        kumo_upper = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        kumo_lower = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Look for entry signals - require: Kumo breakout + volume spike + Kumo twist alignment
            long_breakout = curr_high > kumo_upper
            short_breakout = curr_low < kumo_lower
            
            # Kumo twist filter: only trade in direction of cloud twist
            long_entry = long_breakout and volume_spike[i] and (kumo_twist_aligned[i] > 0.5)
            short_entry = short_breakout and volume_spike[i] and (kumo_twist_aligned[i] < 0.5)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price re-enters Kumo or Kumo twist turns bearish
            if curr_close < kumo_upper or kumo_twist_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price re-enters Kumo or Kumo twist turns bullish
            if curr_close > kumo_lower or kumo_twist_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0