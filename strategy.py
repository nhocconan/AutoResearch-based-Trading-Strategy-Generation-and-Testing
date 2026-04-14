#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout + Daily Volume Spike + ADX Trend Filter
# Uses 12h timeframe with 1d Camarilla pivot levels for entry, volume confirmation for strength,
# and ADX trend filter to avoid false breakouts. Targets 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear by trading breakouts in the direction of the trend using institutional pivot levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ADX (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_h2 = close_1d + (range_1d * 1.1 / 6)
    camarilla_h1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_l1 = close_1d - (range_1d * 1.1 / 12)
    camarilla_l2 = close_1d - (range_1d * 1.1 / 6)
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (wait for 1d close)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    h2_12h = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    h1_12h = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_12h = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    l2_12h = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d ADX for trend filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 2x average volume (20-period) on 12h
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for ADX calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or
            np.isnan(adx_12h[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        if adx_12h[i] < 25:
            # In weak trend/ranging market, stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 Camarilla level with volume filter
            if price > h3_12h[i] and vol > 2.0 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below L3 Camarilla level with volume filter
            elif price < l3_12h[i] and vol > 2.0 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below H1 Camarilla level (take profit) or L4 (stop)
            if price < h1_12h[i] or price < l4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above L1 Camarilla level (take profit) or H4 (stop)
            if price > l1_12h[i] or price > h4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0