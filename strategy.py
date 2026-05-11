#!/usr/bin/env python3
# Hypothesis: 4h Williams %R with volume confirmation and ADX trend filter.
# Williams %R identifies overbought/oversold conditions; reversals from extremes
# work in both bull and bear markets when combined with trend strength (ADX) and volume.
# Uses 1d Williams %R for higher reliability, aligned to 4h.
# Target: 20-40 trades/year per symbol, low frequency to avoid fee drag.
# Entry: Williams %R crosses above/below oversold/overbought with volume spike and ADX > 20.
# Exit: Opposite Williams %R cross or ADX weakening.
# Position size: 0.25.

name = "4h_WilliamsR_Volume_ADX"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        (highest_high - close_1d) / (highest_high - lowest_low) * -100,
        -50.0  # neutral when range is zero
    )
    
    # ADX (14-period) for trend strength
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Plus Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    # Minus Directional Movement
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    # Smoothed TR, +DM, -DM
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    # Directional Indicators
    plus_di = np.where(tr_smooth != 0, plus_dm_smooth / tr_smooth * 100, 0)
    minus_di = np.where(tr_smooth != 0, minus_dm_smooth / tr_smooth * 100, 0)
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(14, 14, 20)  # Williams %R, ADX, volume MA
    
    for i in range(start_idx, n):
        if np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) + volume spike + ADX > 20
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                vol_spike[i] and adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) + volume spike + ADX > 20
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  vol_spike[i] and adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -20 (overbought) or ADX weakens
            if (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -80 (oversold) or ADX weakens
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals