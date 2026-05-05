#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme reversal with volume spike and ADX trend filter
# Long when Williams %R(14) crosses above -80 from below AND ADX(14) > 25 AND volume > 2.0 * avg_volume(20) on 4h
# Short when Williams %R(14) crosses below -20 from above AND ADX(14) > 25 AND volume > 2.0 * avg_volume(20) on 4h
# Exit when Williams %R crosses back through -50 midpoint
# Uses discrete sizing 0.25 to balance return and risk
# Williams %R identifies overbought/oversold conditions that work in both bull and bear markets
# ADX filter ensures we only trade in trending conditions, reducing whipsaw
# Volume confirmation validates reversal strength
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_1dWilliamsR_ADX25_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least one completed daily bar for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14))
    
    # Align 1d Williams %R to 4h timeframe (wait for completed daily bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate ADX(14) on 4h timeframe
    # ADX calculation requires +DI, -DI, and TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Williams %R crossover signals
            williams_r_prev = williams_r_aligned[i-1] if i > 0 else williams_r_aligned[i]
            
            # Long: Williams %R crosses above -80 from below, ADX > 25, volume confirmation, in session
            if (williams_r_prev <= -80 and williams_r_aligned[i] > -80 and adx[i] > 25 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above, ADX > 25, volume confirmation, in session
            elif (williams_r_prev >= -20 and williams_r_aligned[i] < -20 and adx[i] > 25 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back below -50
            williams_r_prev = williams_r_aligned[i-1] if i > 0 else williams_r_aligned[i]
            if williams_r_prev >= -50 and williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back above -50
            williams_r_prev = williams_r_aligned[i-1] if i > 0 else williams_r_aligned[i]
            if williams_r_prev <= -50 and williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals