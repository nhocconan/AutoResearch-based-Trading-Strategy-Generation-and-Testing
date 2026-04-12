#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d volume spike and chop regime filter
    # Donchian channels from 12h provide clear breakout levels
    # Volume spike confirms institutional participation
    # Chop regime filter avoids whipsaws in ranging markets
    # Works in bull/bear by fading extremes in range and following breakouts in trend
    # Target: 12-37 trades/year per symbol (50-150 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    upper_20 = np.full(len(df_12h), np.nan)
    lower_20 = np.full(len(df_12h), np.nan)
    
    for i in range(19, len(df_12h)):
        upper_20[i] = np.max(high_12h[i-19:i+1])
        lower_20[i] = np.min(low_12h[i-19:i+1])
    
    # Align 12h Donchian levels to primary timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # Get 1d data for volume and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume spike filter (current volume > 1.5 * 20-day average)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume > 1.5 * vol_ma_20_1d_aligned
    
    # Chop regime filter (using 1d data)
    # Chop > 61.8 = ranging (mean revert at Donchian upper/lower)
    # Chop < 38.2 = trending (follow breakouts)
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
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if chop_aligned[i] > 61.8:  # Ranging market - mean revert at Donchian levels
            # Long near lower band, short near upper band
            long_entry = close[i] <= lower_aligned[i] and volume_spike[i]
            short_entry = close[i] >= upper_aligned[i] and volume_spike[i]
            long_exit = close[i] >= (upper_aligned[i] + lower_aligned[i]) / 2  # midpoint
            short_exit = close[i] <= (upper_aligned[i] + lower_aligned[i]) / 2  # midpoint
        else:  # Trending market - follow breakouts
            # Breakout entries
            long_entry = close[i] > upper_aligned[i] and volume_spike[i]
            short_entry = close[i] < lower_aligned[i] and volume_spike[i]
            # Exit on opposite test of bands or volume dropout
            long_exit = close[i] < lower_aligned[i] or (not volume_spike[i])
            short_exit = close[i] > upper_aligned[i] or (not volume_spike[i])
        
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

name = "12h_1d_donchian_breakout_vol_chop_v1"
timeframe = "12h"
leverage = 1.0