#!/usr/bin/env python3
# Hypothesis: 4h Williams %R mean reversion with 1d ADX trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) in ADX uptrend with volume > 1.5x average
# Short when Williams %R > -20 (overbought) in ADX downtrend with volume > 1.5x average
# Exit when Williams %R crosses above -50 (long) or below -50 (short)
# Williams %R identifies overbought/oversold conditions, ADX filters trend strength, volume confirms conviction
# Designed to work in both trending and ranging markets with controlled frequency
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "4h_WilliamsR_MeanRev_1dADX_Trend_Volume"
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
    
    # True Range calculation
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = df_1d['high'] - df_1d['high'].shift(1)
    dm_minus = df_1d['low'].shift(1) - df_1d['low']
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smooth TR and DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_rising = adx_values > np.roll(adx_values, 1)
    adx_falling = adx_values < np.roll(adx_values, 1)
    
    # Align ADX and trend signals to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    adx_rising_aligned = align_htf_to_ltf(prices, df_1d, adx_rising)
    adx_falling_aligned = align_htf_to_ltf(prices, df_1d, adx_falling)
    
    # Calculate Williams %R on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(adx_rising_aligned[i]) or 
            np.isnan(adx_falling_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80), ADX rising, volume confirmation
            if (williams_r[i] < -80 and 
                adx_rising_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20), ADX falling, volume confirmation
            elif (williams_r[i] > -20 and 
                  adx_falling_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals