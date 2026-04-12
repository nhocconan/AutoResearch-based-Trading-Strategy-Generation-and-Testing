#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
    # Uses proven Camarilla levels from 1d as strong support/resistance
    # Volume spike confirms breakout validity
    # Chop regime filter adapts: mean revert in range, follow breakouts in trend
    # Target: 20-50 trades/year per symbol (80-200 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h2 = np.full(len(df_1d), np.nan)
    camarilla_l2 = np.full(len(df_1d), np.nan)
    camarilla_h1 = np.full(len(df_1d), np.nan)
    camarilla_l1 = np.full(len(df_1d), np.nan)
    camarilla_pivot = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Use previous day's OHLC
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        
        # Camarilla calculations
        range_prev = high_prev - low_prev
        camarilla_pivot[i] = (high_prev + low_prev + close_prev) / 3
        camarilla_h1[i] = camarilla_pivot[i] + (range_prev * 1.1 / 12)
        camarilla_l1[i] = camarilla_pivot[i] - (range_prev * 1.1 / 12)
        camarilla_h2[i] = camarilla_pivot[i] + (range_prev * 1.1 / 6)
        camarilla_l2[i] = camarilla_pivot[i] - (range_prev * 1.1 / 6)
        camarilla_h3[i] = camarilla_pivot[i] + (range_prev * 1.1 / 4)
        camarilla_l3[i] = camarilla_pivot[i] - (range_prev * 1.1 / 4)
        camarilla_h4[i] = camarilla_pivot[i] + (range_prev * 1.1 / 2)
        camarilla_l4[i] = camarilla_pivot[i] - (range_prev * 1.1 / 2)
    
    # Align 1d Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # 1d volume spike filter (current volume > 1.8 * 20-day average)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume > 1.8 * vol_ma_20_1d_aligned
    
    # Chop regime filter (using 1d data)
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        tr = np.max([
            high_1d[i] - low_1d[i],
            np.abs(high_1d[i] - close_1d[i-1]),
            np.abs(low_1d[i] - close_1d[i-1])
        ])
        if i == 14:
            atr_1d[i] = np.mean([high_1d[j] - low_1d[j] for j in range(i-13, i+1)])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14
    
    # Calculate Chop (14-period)
    sum_tr_14 = np.full(len(df_1d), np.nan)
    max_min_range_14 = np.full(len(df_1d), np.nan)
    chop = np.full(len(df_1d), 50.0)  # default neutral
    
    for i in range(13, len(df_1d)):
        # Sum of true range over 14 periods
        tr_sum = 0
        for j in range(i-13, i+1):
            tr = np.max([
                high_1d[j] - low_1d[j],
                np.abs(high_1d[j] - close_1d[j-1]) if j > 0 else 0,
                np.abs(low_1d[j] - close_1d[j-1]) if j > 0 else 0
            ])
            tr_sum += tr
        sum_tr_14[i] = tr_sum
        
        # Max high - min low over 14 periods
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        max_min_range_14[i] = max_high - min_low
        
        if max_min_range_14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr_14[i] / max_min_range_14[i]) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if chop_aligned[i] > 61.8:  # Ranging market - mean revert at Camarilla H3/L3
            # Long near L3, short near H3
            long_entry = close[i] <= l3_aligned[i] and volume_spike[i]
            short_entry = close[i] >= h3_aligned[i] and volume_spike[i]
            long_exit = close[i] >= pivot_aligned[i]
            short_exit = close[i] <= pivot_aligned[i]
        else:  # Trending market - follow breakouts at H4/L4
            # Breakout entries
            long_entry = close[i] > h4_aligned[i] and volume_spike[i]
            short_entry = close[i] < l4_aligned[i] and volume_spike[i]
            # Exit on opposite test of H4/L4 or volume dropout
            long_exit = close[i] < l4_aligned[i] or (not volume_spike[i])
            short_exit = close[i] > h4_aligned[i] or (not volume_spike[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_vol_chop_v1"
timeframe = "4h"
leverage = 1.0