#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume confirmation.
# Williams %R measures overbought/oversold conditions. Long when %R crosses above -80 from below (oversold bounce)
# in an uptrend (1d ADX > 25 and +DI > -DI). Short when %R crosses below -20 from above (overbought rejection)
# in a downtrend (1d ADX > 25 and -DI > +DI). Volume filter requires 6h volume > 1.5x 20-period average.
# Exit on opposite %R crossover or trend weakening (ADX < 20). Uses 6h timeframe for lower frequency.
# Williams %R is effective in ranging markets; ADX filter ensures we only trade in strong trends.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via buying dips, bear via selling rallies.

name = "6h_WilliamsR_1dADX_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Williams %R(14) on 6h data
    highest_high_6h = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - close_6h) / (highest_high_6h - lowest_low_6h)
    # Handle division by zero (when high == low)
    williams_r_6h[highest_high_6h == lowest_low_6h] = -50
    
    # Volume filter: current 6h volume > 1.5x 20-period average
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume_6h > (1.5 * vol_ma_6h)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Handle division by zero in DX calculation
    dx_1d[plus_di_1d + minus_di_1d == 0] = 0
    adx_1d[np.isnan(adx_1d)] = 0
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN or invalid
        if (np.isnan(williams_r_6h[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(plus_di_1d_aligned[i]) or np.isnan(minus_di_1d_aligned[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below AND ADX > 25 AND +DI > -DI AND volume confirmation
            if (williams_r_6h[i] > -80 and williams_r_6h[i-1] <= -80 and  # crossover above -80
                adx_1d_aligned[i] > 25 and 
                plus_di_1d_aligned[i] > minus_di_1d_aligned[i] and
                volume_filter_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above AND ADX > 25 AND -DI > +DI AND volume confirmation
            elif (williams_r_6h[i] < -20 and williams_r_6h[i-1] >= -20 and  # crossover below -20
                  adx_1d_aligned[i] > 25 and 
                  minus_di_1d_aligned[i] > plus_di_1d_aligned[i] and
                  volume_filter_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -50 OR trend weakening (ADX < 20) OR -DI > +DI
            if (williams_r_6h[i] < -50 and williams_r_6h[i-1] >= -50) or \
               adx_1d_aligned[i] < 20 or \
               minus_di_1d_aligned[i] > plus_di_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -50 OR trend weakening (ADX < 20) OR +DI > -DI
            if (williams_r_6h[i] > -50 and williams_r_6h[i-1] <= -50) or \
               adx_1d_aligned[i] < 20 or \
               plus_di_1d_aligned[i] > minus_di_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals