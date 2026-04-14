#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume confirmation and ADX trend filter
# Williams %R identifies overbought/oversold conditions. In trending markets (ADX>25),
# we fade extreme readings when volume confirms institutional participation.
# Long: Williams %R < -80 (oversold) + volume spike + ADX>25
# Short: Williams %R > -20 (overbought) + volume spike + ADX>25
# Exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Designed to work in both bull and bear markets by fading extremes in trending conditions
# Target: 25-50 trades per symbol over 4 years (6-12.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Williams %R (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (hh - close_4h) / (hh - ll)
    # Handle division by zero when hh == ll
    williams_r = np.where((hh - ll) == 0, -50, williams_r)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for Williams %R and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: Williams %R oversold (< -80) with volume spike and strong trend
            if (williams_r_aligned[i] < -80 and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                adx_aligned[i] > 25):                           # Strong trend
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R overbought (> -20) with volume spike and strong trend
            elif (williams_r_aligned[i] > -20 and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                  adx_aligned[i] > 25):                           # Strong trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (no longer oversold)
            if williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (no longer overbought)
            if williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_MeanReversion_Volume_ADX"
timeframe = "4h"
leverage = 1.0