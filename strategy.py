#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h EMA crossover + 1d ADX trend filter + volume confirmation
# EMA(21) and EMA(55) for trend direction
# 1d ADX(14) > 25 to filter strong trends (avoid chop)
# Volume > 1.5x 20-period average for conviction
# Long when fast EMA > slow EMA, ADX > 25, volume filter
# Short when fast EMA < slow EMA, ADX > 25, volume filter
# Exit when EMA crossover reverses
# Designed to capture strong trends with volume confirmation in both bull and bear markets
# Target: 25-40 trades/year to avoid fee drag
name = "4h_EMA_ADX_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX calculation
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = df_1d['high'] - df_1d['high'].shift(1)
    dm_minus = df_1d['low'].shift(1) - df_1d['low']
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean()
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # 4x EMA calculations
    ema_fast = pd.Series(close).ewm(span=21, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=55, adjust=False).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: EMA crossover up + ADX > 25 + volume
            if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1] and adx_1d_aligned[i] > 25 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: EMA crossover down + ADX > 25 + volume
            elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1] and adx_1d_aligned[i] > 25 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA crossover down
            if ema_fast[i] < ema_slow[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA crossover up
            if ema_fast[i] > ema_slow[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals