#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x average
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short)
# Uses 6h timeframe for lower trade frequency, Williams %R for momentum extremes, 1d ADX for trend filter, volume spike for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via trend continuation, bear via faded rallies.

name = "6h_WilliamsR_ADX_Trend_Volume_v2"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate Williams %R on 6h data (14-period)
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero (when highest_high == lowest_low)
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full_like(close, np.nan)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    if len(high_1d) >= 14:
        # True Range
        tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
        tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
        tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Movement
        up_move = pd.Series(high_1d).diff()
        down_move = -pd.Series(low_1d).diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
        # Handle division by zero (when plus_di + minus_di == 0)
        adx = np.where((plus_di + minus_di) == 0, 0, adx)
    else:
        adx = np.full_like(close_1d, np.nan)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current 6h volume > 1.5x 20-period average (spike confirmation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for Williams %R and ADX
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if williams_r[i] < -80 and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif williams_r[i] > -20 and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (momentum weakening)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (momentum weakening)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals