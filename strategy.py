#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 12h ADX Trend Filter + Volume Spike
- Long when Williams %R < -80 (oversold) AND 12h ADX > 25 (trending) AND volume > 1.5x 20-period average
- Short when Williams %R > -20 (overbought) AND 12h ADX > 25 (trending) AND volume > 1.5x 20-period average
- Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
- Uses 12h ADX for HTF trend alignment to ensure we trade with the higher timeframe trend
- Williams %R identifies extreme reversals within the trend
- Volume spike confirms momentum behind the move
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - pd.Series(close_12h).shift(1)))
    tr3 = pd.Series(np.abs(low_12h - pd.Series(close_12h).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_12h - high_12h.shift(1))
    down_move = pd.Series(low_12h.shift(1) - low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di_12h = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_12h
    minus_di_12h = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_12h
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = dx.rolling(window=14, min_periods=14).mean()
    adx_12h_values = adx_12h.values
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_values)
    
    # Calculate Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r_values = williams_r.values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # Need 14 for Williams %R, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(williams_r_values[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        oversold = williams_r_values[i] < -80
        overbought = williams_r_values[i] > -20
        exit_long = williams_r_values[i] > -50  # Exit long when crosses above -50
        exit_short = williams_r_values[i] < -50  # Exit short when crosses below -50
        
        # Trend filter (using 12h ADX)
        trending = adx_12h_aligned[i] > 25
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R oversold + trending + volume confirmation
            if oversold and trending and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought + trending + volume confirmation
            elif overbought and trending and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses back above/below -50
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if exit_long:
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R crosses below -50
                if exit_short:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hADX_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0