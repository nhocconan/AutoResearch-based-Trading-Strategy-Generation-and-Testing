#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w volume spike and chop regime filter
    # Donchian channels from 1w provide major structural breakout levels
    # Volume spike confirms institutional participation on weekly timeframe
    # Chop regime filter avoids whipsaws in ranging markets
    # Works in bull/bear by fading extremes in range and following breakouts in trend
    # Target: 7-25 trades/year per symbol (30-100 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels and volume context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Donchian channels (20-period)
    donchian_h = np.full(len(df_1w), np.nan)
    donchian_l = np.full(len(df_1w), np.nan)
    donchian_m = np.full(len(df_1w), np.nan)
    
    for i in range(19, len(df_1w)):
        donchian_h[i] = np.max(high_1w[i-19:i+1])
        donchian_l[i] = np.min(low_1w[i-19:i+1])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # Align 1w Donchian channels to 1d timeframe
    h_aligned = align_htf_to_ltf(prices, df_1w, donchian_h)
    l_aligned = align_htf_to_ltf(prices, df_1w, donchian_l)
    m_aligned = align_htf_to_ltf(prices, df_1w, donchian_m)
    
    # 1w volume spike filter (current volume > 1.5 * 20-week average)
    vol_ma_20_1w = np.full(len(df_1w), np.nan)
    for i in range(19, len(df_1w)):
        vol_ma_20_1w[i] = np.mean(volume_1w[i-19:i+1])
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    volume_spike = volume > 1.5 * vol_ma_20_1w_aligned
    
    # Chop regime filter (using 1w data)
    # Chop > 61.8 = ranging (mean revert at Donchian mid)
    # Chop < 38.2 = trending (follow breakouts at H/L)
    atr_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        tr = np.max([
            high_1w[i] - low_1w[i],
            np.abs(high_1w[i] - close_1w[i-1]),
            np.abs(low_1w[i] - close_1w[i-1])
        ])
        if i == 14:
            atr_1w[i] = np.mean([high_1w[j] - low_1w[j] for j in range(i-13, i+1)])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr) / 14
    
    # Calculate Chop (14-period)
    sum_tr_14 = np.full(len(df_1w), np.nan)
    max_min_range_14 = np.full(len(df_1w), np.nan)
    chop = np.full(len(df_1w), 50.0)  # default neutral
    
    for i in range(13, len(df_1w)):
        # Sum of true range over 14 periods
        tr_sum = 0
        for j in range(i-13, i+1):
            tr = np.max([
                high_1w[j] - low_1w[j],
                np.abs(high_1w[j] - close_1w[j-1]) if j > 0 else 0,
                np.abs(low_1w[j] - close_1w[j-1]) if j > 0 else 0
            ])
            tr_sum += tr
        sum_tr_14[i] = tr_sum
        
        # Max high - min low over 14 periods
        max_high = np.max(high_1w[i-13:i+1])
        min_low = np.min(low_1w[i-13:i+1])
        max_min_range_14[i] = max_high - min_low
        
        if max_min_range_14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr_14[i] / max_min_range_14[i]) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h_aligned[i]) or np.isnan(l_aligned[i]) or 
            np.isnan(m_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if chop_aligned[i] > 61.8:  # Ranging market - mean revert at Donchian mid
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