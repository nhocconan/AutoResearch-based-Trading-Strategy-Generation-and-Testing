#!/usr/bin/env python3

"""
12h Weekly Donchian Breakout with Weekly ADX Trend Filter and Volume Confirmation.
Trades breakouts above weekly Donchian high (long) or below weekly Donchian low (short) only when weekly ADX > 25 (trending market).
Uses volume spike to confirm breakout strength. Designed for low trade frequency (12-37/year) to minimize fee drift.
Works in both bull and bear markets by only trading in strong trends (ADX filter) and using volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.diff(high)
    minus_dm = -np.diff(low)
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = np.abs(np.diff(high))
    tr2 = np.abs(np.diff(low))
    tr3 = np.abs(np.diff(close))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter and Donchian channels - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly ADX for trend filter (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Weekly Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_14_1w_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_14_1w_aligned[i] > 25
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and strong_trend and vol_spike:
            # Long: price breaks above weekly Donchian high
            if close[i] > donchian_high_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low
            elif close[i] < donchian_low_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or trend weakens
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below weekly Donchian low or ADX drops below 20
                if close[i] < donchian_low_20_aligned[i] or adx_14_1w_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above weekly Donchian high or ADX drops below 20
                if close[i] > donchian_high_20_aligned[i] or adx_14_1w_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Weekly_Donchian_Breakout_1wADX14_Volume"
timeframe = "12h"
leverage = 1.0