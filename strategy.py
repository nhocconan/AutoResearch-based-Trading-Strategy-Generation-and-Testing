#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and chop regime filter
    # Donchian channels from 1w provide major trend structure
    # Volume spike on 1d confirms institutional participation at breakout
    # Chop regime filter on 1d avoids whipsaws in ranging markets
    # Works in bull/bear by following breakouts in trend and fading extremes in range
    # Target: 7-25 trades/year per symbol (30-100 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels (primary trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channels (20-period)
    donchian_h_1w = np.full(len(df_1w), np.nan)
    donchian_l_1w = np.full(len(df_1w), np.nan)
    donchian_m_1w = np.full(len(df_1w), np.nan)
    
    for i in range(19, len(df_1w)):
        donchian_h_1w[i] = np.max(high_1w[i-19:i+1])
        donchian_l_1w[i] = np.min(low_1w[i-19:i+1])
        donchian_m_1w[i] = (donchian_h_1w[i] + donchian_l_1w[i]) / 2
    
    # Align 1w Donchian channels to 1d timeframe
    h_aligned = align_htf_to_ltf(prices, df_1w, donchian_h_1w)
    l_aligned = align_htf_to_ltf(prices, df_1w, donchian_l_1w)
    m_aligned = align_htf_to_ltf(prices, df_1w, donchian_m_1w)
    
    # Get 1d data for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d volume spike filter (current volume > 2.0 * 20-day average)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume > 2.0 * vol_ma_20_1d_aligned
    
    # Chop regime filter (using 1d data, 14-period)
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
        # Skip if data not ready
        if (np.isnan(h_aligned[i]) or np.isnan(l_aligned[i]) or 
            np.isnan(m_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if chop_aligned[i] > 61.8:  # Ranging market - fade extremes at Donchian bands
            # Long near lower band, short near upper band
            long_entry = close[i] <= l_aligned[i] and volume_spike[i]
            short_entry = close[i] >= h_aligned[i] and volume_spike[i]
            long_exit = close[i] >= m_aligned[i]
            short_exit = close[i] <= m_aligned[i]
        else:  # Trending market - follow breakouts at H/L
            # Breakout entries
            long_entry = close[i] > h_aligned[i] and volume_spike[i]
            short_entry = close[i] < l_aligned[i] and volume_spike[i]
            # Exit on opposite test of H/L or volume dropout
            long_exit = close[i] < l_aligned[i] or (not volume_spike[i])
            short_exit = close[i] > h_aligned[i] or (not volume_spike[i])
        
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

name = "1d_1w_donchian_breakout_vol_chop_v1"
timeframe = "1d"
leverage = 1.0