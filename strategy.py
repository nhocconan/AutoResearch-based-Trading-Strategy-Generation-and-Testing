#!/usr/bin/env python3
"""
Hypothesis:
This strategy combines 6h Donchian breakouts with 1-day pivot direction and volume confirmation.
In trending markets (ADX > 25), price tends to continue in the direction of breakouts from
Donchian channels. The 1-day pivot provides institutional reference points: breaks above
R1 suggest bullish continuation, breaks below S1 suggest bearish continuation. Volume
confirmation ensures the breakout has conviction. Designed for 6h timeframe to achieve
12-37 trades/year with low decay. Works in both bull and bear markets by filtering for
trending regimes and using pivot levels as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, window):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    return upper, lower

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Donchian Channel (20-period) ===
    donch_hi, donch_lo = calculate_donchian(high, low, 20)
    
    # === 1-day Pivot Points (for direction) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points for each 1d bar
    pivot_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        p, r1, s1, _, _, _, _ = calculate_pivot_points(high_1d[i], low_1d[i], close_1d[i])
        pivot_1d[i] = p
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1-day ADX (14-period) for trend filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1-day Volume Spike (vs 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume spike: current 1d volume > 1.5x 20-period average
        vol_spike = volume_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Trend filter: ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_hi[i-1]  # Break above upper band
        breakout_down = close[i] < donch_lo[i-1]  # Break below lower band
        
        # Pivot direction: price relative to R1/S1
        above_r1 = close[i] > r1_aligned[i]
        below_s1 = close[i] < s1_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_spike and trend_filter:
                # Long: breakout up AND above R1 (bullish confluence)
                if breakout_up and above_r1:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: breakout down AND below S1 (bearish confluence)
                elif breakout_down and below_s1:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic: exit when price returns to Donchian mid-point or conditions fail
        elif position == 1:
            donch_mid = (donch_hi[i] + donch_lo[i]) / 2
            # Exit long if price falls below midpoint or conditions fail
            if close[i] < donch_mid or not vol_spike or not trend_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            donch_mid = (donch_hi[i] + donch_lo[i]) / 2
            # Exit short if price rises above midpoint or conditions fail
            if close[i] > donch_mid or not vol_spike or not trend_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dPivotR1S1_Volume1.5x_ADX25"
timeframe = "6h"
leverage = 1.0