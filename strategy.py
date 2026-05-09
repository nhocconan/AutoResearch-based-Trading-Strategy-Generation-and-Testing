#!/usr/bin/env python3
# Hypothesis: 12h Donchian breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high with ADX > 25 and volume > 1.5x average
# Short when price breaks below 20-period Donchian low with ADX > 25 and volume > 1.5x average
# Exit when price crosses the Donchian midline or reverses to opposite Donchian band
# Uses Donchian channels for breakout signals, ADX for trend strength, volume for conviction
# Designed to capture strong momentum moves while avoiding choppy markets
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25

name = "12h_Donchian20_1dADX25_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian high and low
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    
    # Calculate 1d ADX for trend strength filter
    df_1d_copy = df_1d.copy()
    df_1d_copy['high'] = df_1d['high'].values
    df_1d_copy['low'] = df_1d['low'].values
    df_1d_copy['close'] = df_1d['close'].values
    
    # True Range
    tr1 = df_1d_copy['high'] - df_1d_copy['low']
    tr2 = abs(df_1d_copy['high'] - df_1d_copy['close'].shift(1))
    tr3 = abs(df_1d_copy['low'] - df_1d_copy['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Directional Movement
    up_move = df_1d_copy['high'] - df_1d_copy['high'].shift(1)
    down_move = df_1d_copy['low'].shift(1) - df_1d_copy['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean()
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean() / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean() / tr_14
    dx = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = dx.ewm(span=14, adjust=False).mean()
    
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, ADX > 25, volume spike
            if (close[i] > donch_high_aligned[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, ADX > 25, volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses midline or reverses to Donchian low
            if (close[i] < donch_mid_aligned[i]) or (close[i] < donch_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses midline or reverses to Donchian high
            if (close[i] > donch_mid_aligned[i]) or (close[i] > donch_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals