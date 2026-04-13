#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h volume spike confirmation and 1d chop regime filter
    # Long: price > upper Donchian(20) AND volume > 1.5x 20-period avg AND chop < 61.8 (trending)
    # Short: price < lower Donchian(20) AND volume > 1.5x 20-period avg AND chop < 61.8 (trending)
    # Exit: price crosses middle Donchian (20-period midpoint) OR chop > 61.8 (ranging)
    # Using 4h primary timeframe for balance of signal quality and trade frequency.
    # Donchian provides objective breakout levels, volume confirms conviction,
    # chop filter avoids whipsaws in ranging markets. Discrete sizing 0.25 minimizes fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation (more reliable than 4h volume alone)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h volume MA(20) for spike detection
    vol_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(vol_12h), np.nan)
    for i in range(20, len(vol_12h)):
        vol_ma_12h[i] = np.mean(vol_12h[i-20:i])
    vol_spike_12h = vol_12h > (1.5 * vol_ma_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Calculate 1d Chopiness Index(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close_1d index
    
    # ATR(14) - sum of TR over 14 periods
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(atr_1d)):
        atr_1d[i] = np.nansum(tr[i-13:i+1])  # 14 periods including current
    
    # Sum of absolute close changes over 14 periods
    close_chg = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_close_chg_14 = np.full(len(close_1d), np.nan)
    for i in range(14, len(sum_close_chg_14)):
        sum_close_chg_14[i] = np.nansum(close_chg[i-13:i+1])  # 14 periods
    
    # Chopiness Index = 100 * log10(sum_close_chg_14 / atr_1d) / log10(14)
    chop_1d = np.full(len(close_1d), np.nan)
    mask = (atr_1d > 0) & (~np.isnan(atr_1d)) & (~np.isnan(sum_close_chg_14))
    chop_1d[mask] = 100 * np.log10(sum_close_chg_14[mask] / atr_1d[mask]) / np.log10(14)
    
    # Trending regime: chop < 61.8
    trending_regime = chop_1d < 61.8
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime)
    
    # Calculate Donchian(20) channels on 4h data
    # Upper: max(high, 20)
    # Lower: min(low, 20)
    # Middle: (upper + lower) / 2
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-19:i+1])  # 20 periods including current
        lowest_low[i] = np.min(low[i-19:i+1])
    
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Breakout conditions
    long_breakout = close > donchian_upper
    short_breakout = close < donchian_lower
    
    # Exit when price crosses middle line
    long_exit = close < donchian_middle
    short_exit = close > donchian_middle
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for all indicators to be ready
        # Skip if data not ready
        if (np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(trending_regime_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: breakout + volume spike + trending regime
        long_entry = long_breakout[i] and vol_spike_12h_aligned[i] and trending_regime_aligned[i]
        short_entry = short_breakout[i] and vol_spike_12h_aligned[i] and trending_regime_aligned[i]
        
        # Exit conditions: middle line cross OR chop > 61.8 (ranging)
        long_exit_cond = long_exit[i] or (not trending_regime_aligned[i])
        short_exit_cond = short_exit[i] or (not trending_regime_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit_cond:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit_cond:
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

name = "4h_12h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0