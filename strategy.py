#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with volume confirmation and 1d ADX trend filter
# Bollinger Band squeeze (low volatility) precedes explosive moves. We trade breakouts
# from the squeeze only when volume confirms and higher timeframe trend (1d ADX) supports.
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Target: 50-150 total trades over 4 years.
# Timeframe: 4h, HTF: 1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 4h
    bb_length = 20
    bb_mult = 2.0
    sma = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean()
    std = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std()
    upper = sma + bb_mult * std
    lower = sma - bb_mult * std
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper - lower) / sma
    # Squeeze: BB width below 20-period mean of BB width (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean()
    squeeze = bb_width < bb_width_ma.values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(close_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(close_1d, 1)), 
                        np.maximum(np.roll(close_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(squeeze[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: price breaks above upper BB + was in squeeze + volume confirmation + ADX > 25
        if (close[i] > upper[i] and squeeze[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower BB + was in squeeze + volume confirmation + ADX > 25
        elif (close[i] < lower[i] and squeeze[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse BB breakout or ADX < 20 (ranging market) or volatility expansion (end of squeeze)
        elif position == 1 and (close[i] < lower[i] or adx_aligned[i] < 20 or not squeeze[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > upper[i] or adx_aligned[i] < 20 or not squeeze[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_BB_Squeeze_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0