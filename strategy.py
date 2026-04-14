#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout + Volume Spike + 1d ADX Trend Filter
# Uses Camarilla pivot levels (H4/L4) from daily chart for breakout entries
# Requires volume > 2x average volume and 1d ADX > 25 to ensure strong trend
# Works in bull/bear by trading breakouts in direction of higher timeframe trend
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H4, L4) from previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    camarilla_H4 = df_1d['close'] + 1.5 * (df_1d['high'] - df_1d['low'])
    camarilla_L4 = df_1d['close'] - 1.5 * (df_1d['high'] - df_1d['low'])
    
    # Shift to use previous day's levels (avoid look-ahead)
    camarilla_H4 = camarilla_H4.shift(1)
    camarilla_L4 = camarilla_L4.shift(1)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4.values)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4.values)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 2x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for ADX calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when 1d ADX > 25 (strong trend on higher timeframe)
        if adx_1d_aligned[i] < 25:
            # In weak trend/ranging market on 1d, stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H4 with volume filter
            if price > camarilla_H4_aligned[i] and vol > 2.0 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Camarilla L4 with volume filter
            elif price < camarilla_L4_aligned[i] and vol > 2.0 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Camarilla L4
            if price < camarilla_L4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Camarilla H4
            if price > camarilla_H4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Pivot_Volume_ADX"
timeframe = "4h"
leverage = 1.0