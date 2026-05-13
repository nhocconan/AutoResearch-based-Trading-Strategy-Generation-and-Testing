#!/usr/bin/env python3
# Hypothesis: 6h Williams %R + 1d ADX trend filter + volume spike confirmation.
# Long when Williams %R crosses above -80 (oversold bounce) AND 1d ADX > 25 (trending) AND volume > 1.8x average
# Short when Williams %R crosses below -20 (overbought rejection) AND 1d ADX > 25 AND volume > 1.8x average
# Exit when Williams %R returns to -50 (mean reversion) OR ADX < 20 (trend weakening)
# Uses 6h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with daily trend filter for BTC/ETH resilience.
# Williams %R captures mean reversion extremes; ADX filters for trending markets; volume spike confirms momentum.

name = "6h_WilliamsR_ADX_Trend_Volume_v3"
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
    
    # Calculate Williams %R on 6h data (primary timeframe)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    adx_period = 14
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
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
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current 6h volume > 1.8x 20-period average (spike confirmation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 AND ADX > 25 AND volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 AND ADX > 25 AND volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -50 (mean reversion) OR ADX < 20 (trend weakening)
            if williams_r[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -50 (mean reversion) OR ADX < 20 (trend weakening)
            if williams_r[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals