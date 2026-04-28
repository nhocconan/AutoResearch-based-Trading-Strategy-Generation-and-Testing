#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Donchian(20) breakout with 1d ADX25 trend filter and volume confirmation.
# Enter long when price breaks above weekly Donchian high, 1d ADX > 25 (trending), and volume > 1.5x 20-bar average.
# Enter short when price breaks below weekly Donchian low, 1d ADX > 25 (trending), and volume > 1.5x 20-bar average.
# Exit when price crosses the weekly Donchian midpoint (mean of high/low) or ADX drops below 20.
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 80-160 total trades over 4 years (20-40/year) to avoid excessive fee drag.
# Weekly Donchian provides structural support/resistance; 1d ADX filters for strong trends;
# Volume confirmation reduces false breakouts. Works in both bull (breakouts) and bear (strong downtrends).

name = "6h_WeeklyDonchian20_1dADX25_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian(20) - lookback of 20 weeks
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Rolling max/min for Donchian channels
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high = high_20
    donchian_low = low_20
    donchian_mid = (donchian_high + donchian_low) / 2  # Midpoint for exit
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get daily data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high_1d[0] - low_1d[0]  # First TR
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    dm_plus.iloc[0] = 0
    dm_minus.iloc[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20  # Exit when trend weakens
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Exit conditions: price crosses midpoint or trend weakens
        exit_long = close[i] < donchian_mid_aligned[i] or weak_trend
        exit_short = close[i] > donchian_mid_aligned[i] or weak_trend
        
        # Handle entries and exits
        if breakout_up and strong_trend and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and strong_trend and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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