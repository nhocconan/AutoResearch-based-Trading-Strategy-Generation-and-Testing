#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout + Volume Spike + 1d ADX Trend Filter
# Uses Camarilla pivot levels from daily timeframe for precision entries.
# Only trades when 1d ADX > 25 to ensure we're in a trending market (works in bull/bear).
# Volume confirmation (>2x average) avoids false breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ADX (calculated ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Formula: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    #          H3 = C + 1.0*(H-L), L3 = C - 1.0*(H-L)
    #          H2 = C + 0.5*(H-L), L2 = C - 0.5*(H-L)
    #          H1 = C + 0.25*(H-L), L1 = C - 0.25*(H-L)
    # We'll use H3/L3 for breakouts (stronger levels)
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate the range
    rang = prev_high - prev_low
    
    # Camarilla levels
    H3 = prev_close + 1.0 * rang
    L3 = prev_close - 1.0 * rang
    
    # Align to 4h timeframe (wait for daily candle to close)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate 1d ADX for trend filter
    # True Range
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - np.roll(prev_close, 1))
    tr3 = np.abs(prev_low - np.roll(prev_close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((prev_high - np.roll(prev_high, 1)) > (np.roll(prev_low, 1) - prev_low), 
                       np.maximum(prev_high - np.roll(prev_high, 1), 0), 0)
    dm_minus = np.where((np.roll(prev_low, 1) - prev_low) > (prev_high - np.roll(prev_high, 1)), 
                        np.maximum(np.roll(prev_low, 1) - prev_low, 0), 0)
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
    
    # Align ADX to 4h
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
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when 1d ADX > 25 (trending market)
        if adx_1d_aligned[i] < 25:
            # In weak trend/ranging market, stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 level with volume filter
            if price > H3_aligned[i] and vol > 2.0 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below L3 level with volume filter
            elif price < L3_aligned[i] and vol > 2.0 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L3 level
            if price < L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H3 level
            if price > H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Volume_1dADX_Filter"
timeframe = "4h"
leverage = 1.0