#!/usr/bin/env python3
# Hypothesis: 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above 20-period high, 1d ADX > 25, and volume > 1.5x average
# Short when price breaks below 20-period low, 1d ADX > 25, and volume > 1.5x average
# Exit when price returns to the middle of the Donchian channel or reverses to opposite side
# Uses price channels for breakout detection, ADX for trend strength, volume for conviction
# Designed to capture strong momentum moves in both trending and ranging markets
# Target: 100-180 total trades over 4 years (25-45/year) with size 0.25

name = "4h_Donchian_Breakout_1dADX25_VolumeConfirm"
timeframe = "4h"
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
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_ma
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_ma
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_1d = adx.values
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, ADX > 25, volume confirmation
            if (close[i] > donchian_high[i] and 
                adx_1d_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, ADX > 25, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  adx_1d_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle of channel or breaks below Donchian low
            if (close[i] <= donchian_mid[i]) or (close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle of channel or breaks above Donchian high
            if (close[i] >= donchian_mid[i]) or (close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals