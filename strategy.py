#!/usr/bin/env python3
"""
1d_1w_DonchianBreakout_TrendFilter_Volume
Hypothesis: Uses weekly Donchian breakout for entry, daily ADX for trend strength, and volume confirmation.
Designed for low trade frequency (10-25/year) with strong trend following. Works in both bull and bear markets
by capturing breakouts in the direction of higher-timeframe trend.
"""

name = "1d_1w_DonchianBreakout_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX indicator"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.nan_to_num(dx, nan=0)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx, plus_di, minus_di

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Donchian for Breakout ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    donch_upper_1w, donch_lower_1w = calculate_donchian(
        df_1w['high'].values, df_1w['low'].values, period=20
    )
    
    # Align weekly Donchian to daily timeframe
    donch_upper_1d = align_htf_to_ltf(prices, df_1w, donch_upper_1w)
    donch_lower_1d = align_htf_to_ltf(prices, df_1w, donch_lower_1w)
    
    # --- Daily ADX for Trend Filter ---
    adx_1d, plus_di_1d, minus_di_1d = calculate_adx(
        high, low, close, period=14
    )
    
    # --- Volume Confirmation (20-day average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_upper_1d[i]) or np.isnan(donch_lower_1d[i]) or 
            np.isnan(adx_1d[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d[i] > 25
        
        # Volume confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper + strong trend + volume
            if (close[i] > donch_upper_1d[i] and 
                strong_trend and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower + strong trend + volume
            elif (close[i] < donch_lower_1d[i] and 
                  strong_trend and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to middle of Donchian channel or trend weakens
            donch_middle = (donch_upper_1d[i] + donch_lower_1d[i]) / 2
            weak_trend = adx_1d[i] < 20  # Exit when trend weakens
            
            if position == 1:
                # Exit long: price below middle OR trend weakens
                if close[i] < donch_middle or weak_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price above middle OR trend weakens
                if close[i] > donch_middle or weak_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals