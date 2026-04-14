#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1-day ADX filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND daily ADX > 25 (trending) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND daily ADX > 25 (trending) AND volume > 1.5x average
# Exit when Williams %R crosses -50 in opposite direction
# Williams %R identifies momentum extremes, ADX filters for trending conditions only,
# volume confirms institutional participation. Designed to capture trends in both bull and bear markets.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R on 4h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
    
    # Calculate ADX on 1d (14-period)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get ADX value aligned to 4h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
        adx_val = adx_aligned[i]
        
        williams_val = williams_r[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume confirmation
            if (williams_val < -80 and adx_val > 25 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume confirmation
            elif (williams_val > -20 and adx_val > 25 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if williams_val > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if williams_val < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_ADX_Volume"
timeframe = "4h"
leverage = 1.0