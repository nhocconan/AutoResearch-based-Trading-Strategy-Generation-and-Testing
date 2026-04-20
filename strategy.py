#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: In trending markets, price breaks out of the 20-period Donchian channel with strong volume.
The ADX filter ensures we only trade in trending conditions (ADX > 25), avoiding whipsaws in ranging markets.
Volume confirmation (> 1.5x average) filters out weak breakouts.
This combination works in both bull and bear markets by capturing strong directional moves.
Target: 25-40 trades per year by requiring multiple confirmations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for ADX calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_daily, prepend=high_daily[0])
    down_move = -np.diff(low_daily, prepend=low_daily[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / np.where(tr_14 == 0, 1, tr_14)
    minus_di_14 = 100 * minus_dm_14 / np.where(tr_14 == 0, 1, tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) == 0, 1, (plus_di_14 + minus_di_14))
    adx_daily = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ADX to 4h timeframe
    adx_daily_aligned = align_htf_to_ltf(prices, df_daily, adx_daily)
    
    # Main timeframe data (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(adx_daily_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx = adx_daily_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx > 25
        
        if position == 0:
            # Long breakout: price breaks above upper channel with volume and trend confirmation
            if price > upper_channel and vol_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below lower channel with volume and trend confirmation
            elif price < lower_channel and vol_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower channel or trend weakens
            if price < lower_channel or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper channel or trend weakens
            if price > upper_channel or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0