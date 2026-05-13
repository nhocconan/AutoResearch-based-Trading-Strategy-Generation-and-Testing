#!/usr/bin/env python3
# Hypothesis: 6h strategy using 1d Williams %R for mean reversion in ranging markets and 1d ADX for trend filtering.
# Enters long when 1d Williams %R < -80 (oversold) and 1d ADX < 25 (ranging/weak trend).
# Enters short when 1d Williams %R > -20 (overbought) and 1d ADX < 25.
# Exits when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position sizing (0.25) to minimize fee drag. Designed for low trade frequency (~10-25/year)
# to work in both bull and bear markets by fading extremes in ranging conditions and avoiding strong trends.

name = "6h_WilliamsR_ADX_Ranging_MeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R parameters
    williams_period = 14
    highest_high = pd.Series(high_1d).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate ADX (Average Directional Index)
    adx_period = 14
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    # Handle division by zero
    plus_di = np.where(atr == 0, 0, plus_di)
    minus_di = np.where(atr == 0, 0, minus_di)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.3x 20-period average (milder than 1.5x to allow more trades)
    volume = prices['volume'].values
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for all indicators
        if np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) and ranging market (ADX < 25) with volume spike
            if williams_r_aligned[i] < -80 and adx_aligned[i] < 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20) and ranging market (ADX < 25) with volume spike
            elif williams_r_aligned[i] > -20 and adx_aligned[i] < 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (mean reversion complete)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (mean reversion complete)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals