#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
    # Donchian channels provide objective breakout levels
    # Volume spike confirms institutional participation  
    # Chop regime filter adapts strategy: mean revert in range (H3/L3), follow breakout in trend (H4/L4)
    # Works in bull/bear markets by switching logic based on volatility regime
    # Target: 20-50 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels, volume context, and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_h = np.full(len(df_1d), np.nan)
    donchian_l = np.full(len(df_1d), np.nan)
    donchian_m = np.full(len(df_1d), np.nan)
    
    for i in range(19, len(df_1d)):
        donchian_h[i] = np.max(high_1d[i-19:i+1])
        donchian_l[i] = np.min(low_1d[i-19:i+1])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # Calculate 1d ATR(14) for Chop calculation
    atr_1d = np.full(len(df_1d), np.nan)
    tr_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = np.max([
                high_1d[i] - low_1d[i],
                np.abs(high_1d[i] - close_1d[i-1]),
                np.abs(low_1d[i] - close_1d[i-1])
            ])
    
    for i in range(13, len(df_1d)):
        if i == 13:
            atr_1d[i] = np.mean(tr_1d[0:14])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate Chop (14-period)
    sum_tr_14 = np.full(len(df_1d), np.nan)
    max_min_range_14 = np.full(len(df_1d), np.nan)
    chop = np.full(len(df_1d), 50.0)  # default neutral
    
    for i in range(13, len(df_1d)):
        # Sum of true range over 14 periods
        tr_sum = np.sum(tr_1d[i-13:i+1])
        sum_tr_14[i] = tr_sum
        
        # Max high - min low over 14 periods
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        max_min_range_14[i] = max_high - min_low
        
        if max_min_range_14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr_14[i] / max_min_range_14[i]) / np.log10(14)
    
    # 1d volume spike filter (current volume > 1.5 * 20-day average)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align 1d indicators to 4h timeframe
    donchian_h_aligned = align_htf_to_ltf(prices, df_1d, donchian_h)
    donchian_l_aligned = align_htf_to_ltf(prices, df_1d, donchian_l)
    donchian_m_aligned = align_htf_to_ltf(prices, df_1d, donchian_m)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    volume_spike = volume > 1.5 * vol_ma_20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(donchian_h_aligned[i]) or np.isnan(donchian_l_aligned[i]) or 
            np.isnan(donchian_m_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if chop_aligned[i] > 61.8:  # Ranging market - mean revert at Donchian levels
            # Long near lower band, short near upper band
            long_entry = close[i] <= donchian_l_aligned[i] and volume_spike[i]
            short_entry = close[i] >= donchian_h_aligned[i] and volume_spike[i]
            long_exit = close[i] >= donchian_m_aligned[i]
            short_exit = close[i] <= donchian_m_aligned[i]
        else:  # Trending market - follow breakouts
            # Breakout entries
            long_entry = close[i] > donchian_h_aligned[i] and volume_spike[i]
            short_entry = close[i] < donchian_l_aligned[i] and volume_spike[i]
            # Exit on opposite test of bands or volume dropout
            long_exit = close[i] < donchian_l_aligned[i] or (not volume_spike[i])
            short_exit = close[i] > donchian_h_aligned[i] or (not volume_spike[i])
        
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

name = "4h_1d_donchian_breakout_vol_chop_v1"
timeframe = "4h"
leverage = 1.0