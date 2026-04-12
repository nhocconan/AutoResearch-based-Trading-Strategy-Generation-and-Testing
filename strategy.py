#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h volume spike and chop regime filter
    # Camarilla levels from 4h provide institutional support/resistance
    # Volume spike confirms institutional participation
    # Chop regime filter avoids whipsaws in ranging markets
    # Works in bull/bear by fading extremes in range and following breakouts in trend
    # Target: 15-35 trades/year per symbol (60-150 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and volume context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Camarilla levels (using previous bar's range)
    camarilla_h4 = np.full(len(df_4h), np.nan)
    camarilla_l4 = np.full(len(df_4h), np.nan)
    camarilla_h3 = np.full(len(df_4h), np.nan)
    camarilla_l3 = np.full(len(df_4h), np.nan)
    camarilla_h2 = np.full(len(df_4h), np.nan)
    camarilla_l2 = np.full(len(df_4h), np.nan)
    camarilla_h1 = np.full(len(df_4h), np.nan)
    camarilla_l1 = np.full(len(df_4h), np.nan)
    camarilla_pivot = np.full(len(df_4h), np.nan)
    
    for i in range(1, len(df_4h)):
        # Previous bar's OHLC
        phigh = high_4h[i-1]
        plow = low_4h[i-1]
        pclose = close_4h[i-1]
        
        # Pivot point
        camarilla_pivot[i] = (phigh + plow + 2 * pclose) / 4
        
        # Range
        rng = phigh - plow
        
        # Camarilla levels
        camarilla_h4[i] = camarilla_pivot[i] + rng * 1.1 / 2
        camarilla_l4[i] = camarilla_pivot[i] - rng * 1.1 / 2
        camarilla_h3[i] = camarilla_pivot[i] + rng * 1.1 / 4
        camarilla_l3[i] = camarilla_pivot[i] - rng * 1.1 / 4
        camarilla_h2[i] = camarilla_pivot[i] + rng * 1.1 / 6
        camarilla_l2[i] = camarilla_pivot[i] - rng * 1.1 / 6
        camarilla_h1[i] = camarilla_pivot[i] + rng * 1.1 / 12
        camarilla_l1[i] = camarilla_pivot[i] - rng * 1.1 / 12
    
    # Align 4h Camarilla levels to 1h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l1)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    # 4h volume spike filter (current volume > 1.5 * 20-bar average)
    vol_ma_20_4h = np.full(len(df_4h), np.nan)
    for i in range(19, len(df_4h)):
        vol_ma_20_4h[i] = np.mean(volume_4h[i-19:i+1])
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    volume_spike = volume > 1.5 * vol_ma_20_4h_aligned
    
    # Chop regime filter (using 4h data)
    # Chop > 61.8 = ranging (mean revert at Camarilla H3/L3)
    # Chop < 38.2 = trending (follow breakouts at H4/L4)
    atr_4h = np.full(len(df_4h), np.nan)
    for i in range(14, len(df_4h)):
        tr = np.max([
            high_4h[i] - low_4h[i],
            np.abs(high_4h[i] - close_4h[i-1]),
            np.abs(low_4h[i] - close_4h[i-1])
        ])
        if i == 14:
            atr_4h[i] = np.mean([high_4h[j] - low_4h[j] for j in range(i-13, i+1)])
        else:
            atr_4h[i] = (atr_4h[i-1] * 13 + tr) / 14
    
    # Calculate Chop (14-period)
    sum_tr_14 = np.full(len(df_4h), np.nan)
    max_min_range_14 = np.full(len(df_4h), np.nan)
    chop = np.full(len(df_4h), 50.0)  # default neutral
    
    for i in range(13, len(df_4h)):
        # Sum of true range over 14 periods
        tr_sum = 0
        for j in range(i-13, i+1):
            tr = np.max([
                high_4h[j] - low_4h[j],
                np.abs(high_4h[j] - close_4h[j-1]) if j > 0 else 0,
                np.abs(low_4h[j] - close_4h[j-1]) if j > 0 else 0
            ])
            tr_sum += tr
        sum_tr_14[i] = tr_sum
        
        # Max high - min low over 14 periods
        max_high = np.max(high_4h[i-13:i+1])
        min_low = np.min(low_4h[i-13:i+1])
        max_min_range_14[i] = max_high - min_low
        
        if max_min_range_14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr_14[i] / max_min_range_14[i]) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if chop_aligned[i] > 61.8:  # Ranging market - mean revert at H3/L3
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
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_camarilla_breakout_vol_chop_v1"
timeframe = "1h"
leverage = 1.0