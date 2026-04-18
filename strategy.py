#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA13 EMA55 Crossover with 1d Volume Ratio Filter and ADX Trend Strength
# EMA13/EMA55 crossover provides trend signals on 6h timeframe.
# 1d Volume Ratio (current volume / 20-period average) > 1.5 confirms institutional participation.
# ADX > 25 on 1d ensures we only trade in strong trending markets (avoids chop/range).
# Works in bull markets (uptrend + volume + ADX) and bear markets (downtrend + volume + ADX).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "6h_EMA_Cross_1dVol_ADXFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume ratio and ADX filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 and EMA55 for crossover
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema55 = close_s.ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Calculate 1d volume ratio: current volume / 20-period average volume
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / vol_ma_20_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(dx)] = 0  # Handle division by zero
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # Wait for EMA55 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13[i]) or np.isnan(ema55[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # EMA crossover signals
        ema13_above_ema55 = ema13[i] > ema55[i]
        ema13_below_ema55 = ema13[i] < ema55[i]
        
        # Filter conditions
        strong_volume = vol_ratio_1d_aligned[i] > 1.5
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: EMA13 crosses above EMA55 AND strong volume AND strong trend
            if ema13_above_ema55 and strong_volume and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: EMA13 crosses below EMA55 AND strong volume AND strong trend
            elif ema13_below_ema55 and strong_volume and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA13 crosses below EMA55 OR trend weakens
            if ema13_below_ema55 or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA13 crosses above EMA55 OR trend weakens
            if ema13_above_ema55 or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals