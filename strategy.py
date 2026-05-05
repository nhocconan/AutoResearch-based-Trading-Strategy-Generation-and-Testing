#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d ADX25 trend filter and volume confirmation
# Long when Williams %R(14) < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
# Short when Williams %R(14) > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# Uses 6h primary timeframe with 1d HTF for ADX trend filter and Williams %R calculation
# Williams %R identifies overextended moves ripe for reversal in both bull and bear markets
# ADX filter ensures we only trade in trending conditions where reversals are more meaningful
# Volume confirmation reduces false signals from low-participation moves
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe

name = "6h_WilliamsR_Extreme_1dADX25_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX trend filter and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean()
        
        # Directional Movement
        dm_plus = pd.Series(df_1d['high']).diff()
        dm_minus = -pd.Series(df_1d['low']).diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smoothed DM and TR
        dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).mean()
        dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).mean()
        atr_smooth = atr.rolling(window=14, min_periods=14).mean()
        
        # Directional Indicators
        di_plus = 100 * (dm_plus_smooth / atr_smooth)
        di_minus = 100 * (dm_minus_smooth / atr_smooth)
        
        # ADX
        dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus)).replace([np.inf, -np.inf], 0)
        adx = dx.rolling(window=14, min_periods=14).mean()
        adx_1d = adx.values
    else:
        adx_1d = np.full(len(df_1d), np.nan)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Williams %R for entry signals
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    else:
        williams_r = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume spike
            if (williams_r[i] < -80 and 
                adx_1d_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume spike
            elif (williams_r[i] > -20 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (exiting oversold territory)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (exiting overbought territory)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals