#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX25 Trend Filter + Volume Spike Confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20) signal potential reversals
# ADX > 25 confirms trending market to avoid whipsaws in ranging conditions
# Volume spike (>2.0x 20-period average) confirms conviction behind the move
# Long when %R < -80 (oversold) AND ADX > 25 AND volume spike
# Short when %R > -20 (overbought) AND ADX > 25 AND volume spike
# Exit when %R crosses back above -50 (for longs) or below -50 (for shorts) OR ADX < 20 (trend weakening)
# Uses 6h primary timeframe with 1d HTF for ADX and Williams %R to reduce noise
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Works in both bull and bear markets by focusing on extreme reversals within established trends

name = "6h_WilliamsR_EXTREME_1dADX25_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for Williams %R (14) and ADX (14)
        return np.zeros(n)
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high_14 - df_1d['close']) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = williams_r.fillna(-50).values  # Neutral when no range
    
    # Calculate ADX on 1d
    # True Range
    tr1 = pd.Series(df_1d['high'].values - df_1d['low'].values)
    tr2 = pd.Series(np.abs(df_1d['high'].values - df_1d['close'].shift(1)))
    tr3 = pd.Series(np.abs(df_1d['low'].values - df_1d['close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d['high'].values - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low'].values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    atr_smooth = atr  # Already smoothed
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    # Handle division by zero and NaN
    plus_di = plus_di.fillna(0).values
    minus_di = minus_di.fillna(0).values
    dx = dx.fillna(0).values
    adx = adx.fillna(0).values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND ADX > 25 AND volume spike
            if (williams_r_aligned[i] < -80 and 
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND ADX > 25 AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR ADX < 20 (trend weakening)
            if (williams_r_aligned[i] > -50 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR ADX < 20 (trend weakening)
            if (williams_r_aligned[i] < -50 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals