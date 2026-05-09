#!/usr/bin/env python3
# Hypothesis: 6h ADX + 12h Donchian breakout with volume confirmation
# Long when ADX > 25 (trending), price breaks above 12h Donchian upper (20), and volume > 1.5x average
# Short when ADX > 25, price breaks below 12h Donchian lower (20), and volume > 1.5x average
# Exit when ADX < 20 (range) or price returns to 12h Donchian middle
# Uses ADX for trend strength, Donchian for breakout levels, volume for confirmation
# Designed to capture strong trends while avoiding whipsaws in ranging markets
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_ADX_Donchian_Breakout_12h_Volume"
timeframe = "6h"
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
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Calculate ADX (14-period) on 6h data
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_sum / np.where(tr_sum == 0, 1, tr_sum)
    minus_di = 100 * minus_dm_sum / np.where(tr_sum == 0, 1, tr_sum)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: ADX > 25 (trending), price breaks above Donchian high, volume spike
            if (adx[i] > 25 and 
                close[i] > donchian_high_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 (trending), price breaks below Donchian low, volume spike
            elif (adx[i] > 25 and 
                  close[i] < donchian_low_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: ADX < 20 (range) or price returns to Donchian middle
            if (adx[i] < 20) or (close[i] <= donchian_mid_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: ADX < 20 (range) or price returns to Donchian middle
            if (adx[i] < 20) or (close[i] >= donchian_mid_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals