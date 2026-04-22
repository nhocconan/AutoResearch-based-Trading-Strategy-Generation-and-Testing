#!/usr/bin/env python3
"""
Hypothesis: Daily chart (1d) with weekly (1w) ADX trend filter and Donchian(20) breakout.
Long when price breaks above Donchian(20) high with weekly ADX > 25 (strong trend).
Short when price breaks below Donchian(20) low with weekly ADX > 25.
Exit when price crosses opposite Donchian band or weekly ADX drops below 20 (trend weakening).
Uses daily timeframe for lower trade frequency and weekly ADX for trend strength filtering.
Designed to capture trends while avoiding choppy markets, working in both bull and bear phases.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data for ADX - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(tr)
    minus_di = np.zeros_like(tr)
    
    # Initial values
    atr[13] = np.mean(tr[1:14])
    plus_dm_sum = np.sum(plus_dm[1:14])
    minus_dm_sum = np.sum(minus_dm[1:14])
    
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_sum = plus_dm_sum - (plus_dm_sum/14) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum/14) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    # DX and ADX
    dx = np.zeros_like(tr)
    dx[14:] = 100 * np.abs(plus_di[14:] - minus_di[14:]) / (plus_di[14:] + minus_di[14:])
    dx[plus_di + minus_di == 0] = 0
    
    adx = np.zeros_like(tr)
    adx[27] = np.mean(dx[14:28])
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for Donchian
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with strong weekly trend
            if close[i] > donchian_high[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with strong weekly trend
            elif close[i] < donchian_low[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Donchian low OR trend weakens
                if close[i] < donchian_low[i] or adx_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Donchian high OR trend weakens
                if close[i] > donchian_high[i] or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_ADX_WeeklyDonchianBreakout"
timeframe = "1d"
leverage = 1.0