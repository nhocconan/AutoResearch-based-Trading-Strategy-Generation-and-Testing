#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and chop regime filter.
# Enter long when price breaks above 1d Donchian upper channel and volume > 1.5x 20-bar average and CHOP(14) > 61.8 (range).
# Enter short when price breaks below 1d Donchian lower channel and volume > 1.5x 20-bar average and CHOP(14) > 61.8.
# Exit when price crosses 1d Donchian midpoint or CHOP < 38.2 (trend regime).
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Donchian breakouts capture momentum, volume confirmation adds conviction, chop filter avoids whipsaws in strong trends.

name = "12h_DonchianBreakout_1d_VolumeChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and chop regime
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper: highest high over 20 periods
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low over 20 periods
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of upper and lower
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d Chopiness Index(14) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(atr_sum / (true_range_max - true_range_min)) / log10(14)
    # We'll use a common approximation: CHOP = 100 * log10(sum(tr) / (hh - ll)) / log10(14)
    # where tr = true range, hh = highest high, ll = lowest low over 14 periods
    tr1 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values - pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    tr2 = abs(pd.Series(high_1d).rolling(window=14, min_periods=14).max().values - pd.Series(close_1d).shift(1).rolling(window=14, min_periods=14).min().values)
    tr3 = abs(pd.Series(low_1d).rolling(window=14, min_periods=14).min().values - pd.Series(close_1d).shift(1).rolling(window=14, min_periods=14).max().values)
    true_range = np.maximum.reduce([tr1, tr2, tr3])
    atr_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = hh_14 - ll_14
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_raw = 100 * np.log10(atr_sum / chop_denominator) / np.log10(14)
    chop_values = chop_raw  # Already scaled 0-100
    
    # Align 1d indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Chop regime: >61.8 = range (good for mean reversion/breakouts in range)
        chop_regime = chop_aligned[i] > 61.8
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Exit conditions
        long_exit = close[i] < donchian_mid_aligned[i] or chop_aligned[i] < 38.2
        short_exit = close[i] > donchian_mid_aligned[i] or chop_aligned[i] < 38.2
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and chop_regime
        short_entry = short_breakout and vol_confirm and chop_regime
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals