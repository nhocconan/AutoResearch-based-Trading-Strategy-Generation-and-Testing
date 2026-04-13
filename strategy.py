#!/usr/bin/env python3
"""
1d_1w_Range_Breakout_With_Volume_Confirmation_and_Range_Filter
Hypothesis: Daily breakout above/below weekly Donchian channels with volume confirmation and weekly range filter.
Long when price breaks above weekly Donchian high (20) + daily volume > 1.5x 20-day average + weekly range < 50th percentile (non-trending).
Short when price breaks below weekly Donchian low (20) + daily volume > 1.5x 20-day average + weekly range < 50th percentile.
Exit when price crosses weekly Donchian mid-point or weekly range expands above 70th percentile (trending regime).
Designed for 1d timeframe to target 10-25 trades/year with strong survival in bear markets via range filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high_20 + donch_low_20) / 2
    
    # Align to daily
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid)
    
    # Weekly range for regime filter (ATR-based)
    atr_10 = pd.Series(np.maximum.reduce([
        high_1w - low_1w,
        np.abs(high_1w - np.roll(close_1w, 1)),
        np.abs(low_1w - np.roll(close_1w, 1))
    ])).rolling(window=10, min_periods=10).mean().values
    
    # Percentile rank of ATR over 50 weeks
    atr_percentile = pd.Series(atr_10).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Daily volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or np.isnan(atr_percentile_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current daily volume > 1.5x 20-day average
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # Range condition: weekly ATR percentile < 50 (non-trending/choppy)
        range_condition = atr_percentile_aligned[i] < 50
        
        # Breakout conditions
        long_breakout = close[i] > donch_high_20_aligned[i]
        short_breakout = close[i] < donch_low_20_aligned[i]
        
        # Exit conditions
        long_exit = close[i] < donch_mid_aligned[i]
        short_exit = close[i] > donch_mid_aligned[i]
        range_expansion = atr_percentile_aligned[i] > 70  # trending regime
        
        if position == 0:
            if long_breakout and vol_condition and range_condition:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition and range_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit or range_expansion:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit or range_expansion:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Range_Breakout_With_Volume_Confirmation_and_Range_Filter"
timeframe = "1d"
leverage = 1.0