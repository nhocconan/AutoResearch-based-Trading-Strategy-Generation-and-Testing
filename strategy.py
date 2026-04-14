#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Trend Filter
# Uses Elder Ray (Bull/Bear Power) for entry signals on 6h timeframe
# with 1d ADX > 25 to filter for trending markets only
# Works in both bull and bear markets by aligning with trend direction
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d = adx_14
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Elder Ray (Bull/Bear Power)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for EMA13 and ADX
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        if adx_aligned[i] < 25:
            # In weak trend/ranging market, stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum)
            if bull_power[i] > 0:
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 (bearish momentum)
            elif bear_power[i] < 0:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bear Power < 0 (momentum shifts bearish)
            if bear_power[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bull Power > 0 (momentum shifts bullish)
            if bull_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_1dADX_TrendFilter"
timeframe = "6h"
leverage = 1.0