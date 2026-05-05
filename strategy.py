#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume spike
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5 * 20-period avg volume
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5 * 20-period avg volume
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Williams %R identifies exhaustion points in trending markets; ADX filters for strong trends only;
# Volume spike confirms institutional participation. Works in bull markets via longs on pullbacks
# and bear markets via shorts on rallies within the primary trend. Uses 1d for HTF trend alignment
# and 6h for entry timing to avoid excessive trading.

name = "6h_WilliamsR_EXTREME_1dADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate 6h Williams %R (14-period) based on previous 6h bar
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Shift to use previous bar's levels (signal based on completed 6h bar)
    williams_r = np.roll(williams_r, 1)
    williams_r[0] = np.nan  # First value invalid after roll
    
    # Align Williams %R to prices timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    up_move = np.where(up_move < 0, 0, up_move)
    down_move = np.where(down_move < 0, 0, down_move)
    
    # +DI and -DI
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di_14 = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di_14 = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Shift ADX to use previous bar's value (signal based on completed 1d bar)
    adx_1d = np.roll(adx_1d, 1)
    adx_1d[0] = np.nan
    
    # Align 1d ADX to prices timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike filter: volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND ADX > 25 (strong trend) AND volume spike
            if (williams_r_aligned[i] < -80 and 
                adx_1d_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND ADX > 25 (strong trend) AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (return from oversold) OR ADX < 20 (weakening trend)
            if (williams_r_aligned[i] > -50 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (return from overbought) OR ADX < 20 (weakening trend)
            if (williams_r_aligned[i] < -50 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals