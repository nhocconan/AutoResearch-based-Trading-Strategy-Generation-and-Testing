#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 15-period Exponential Moving Average (EMA15) crossover with 4h 50-period EMA (EMA50) trend filter,
# combined with 4h volume confirmation (volume > 1.5x 20-period average) and 1d ADX regime filter (ADX > 25 for trending).
# Long when EMA15 crosses above EMA50, volume confirms, and 1d ADX > 25.
# Short when EMA15 crosses below EMA50, volume confirms, and 1d ADX > 25.
# Exit when EMA15 crosses back in the opposite direction.
# This strategy targets trending markets with momentum confirmation, avoiding choppy regimes.
# Target: 20-50 trades per year on 4h timeframe for low friction and high signal quality.

name = "4h_EMA15_EMA50_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA15 and EMA50 on 4h data
    ema15 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Calculate ADX on 1d data (14-period)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50 and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema15[i]) or np.isnan(ema50[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: EMA15 crosses above EMA50, volume confirms, ADX > 25 (trending)
            long_cond = (ema15[i] > ema50[i]) and (ema15[i-1] <= ema50[i-1]) and volume_filter[i] and (adx_aligned[i] > 25)
            # Short conditions: EMA15 crosses below EMA50, volume confirms, ADX > 25 (trending)
            short_cond = (ema15[i] < ema50[i]) and (ema15[i-1] >= ema50[i-1]) and volume_filter[i] and (adx_aligned[i] > 25)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: EMA15 crosses back below EMA50
            if ema15[i] < ema50[i] and ema15[i-1] >= ema50[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: EMA15 crosses back above EMA50
            if ema15[i] > ema50[i] and ema15[i-1] <= ema50[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals