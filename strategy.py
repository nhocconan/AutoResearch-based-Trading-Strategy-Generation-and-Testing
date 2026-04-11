#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 1 or len(df_1d) < 20:
        return signals
    
    # Calculate weekly pivot points (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R1, R2, S1, S2
    r1_1w = 2 * pivot_1w - low_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s1_1w = 2 * pivot_1w - high_1w
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Shift to use previous week's data to avoid look-ahead
    pivot_1w = np.roll(pivot_1w, 1)
    r1_1w = np.roll(r1_1w, 1)
    r2_1w = np.roll(r2_1w, 1)
    s1_1w = np.roll(s1_1w, 1)
    s2_1w = np.roll(s2_1w, 1)
    # Set first week to NaN
    pivot_1w[0] = r1_1w[0] = r2_1w[0] = s1_1w[0] = s2_1w[0] = np.nan
    
    # Align weekly pivots to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Daily Donchian breakout for entry timing (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or
            np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        upper_channel = donchian_high_20_aligned[i]
        lower_channel = donchian_low_20_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        # Long: price breaks above weekly R2 with volume confirmation
        if price_high > r2 and volume_confirmed:
            long_signal = True
        
        # Short: price breaks below weekly S2 with volume confirmation
        if price_low < s2 and volume_confirmed:
            short_signal = True
        
        # Exit conditions
        # Exit long when price returns to weekly pivot
        exit_long = position == 1 and price_close < pivot
        # Exit short when price returns to weekly pivot
        exit_short = position == -1 and price_close > pivot
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly pivot breakout strategy with volume confirmation on 6h timeframe.
# Enters long when price breaks above weekly R2 pivot level with volume confirmation (>1.5x avg volume).
# Enters short when price breaks below weekly S2 pivot level with volume confirmation.
# Uses weekly timeframe for pivot points to capture major support/resistance levels.
# Volume confirmation ensures institutional participation and reduces false breakouts.
# Exits when price returns to the weekly pivot point, capturing mean reversion within the week.
# Designed for 6h timeframe with tight entry conditions to target 50-150 total trades over 4 years.
# Works in both bull and bear markets by trading breakouts in either direction from key weekly levels.