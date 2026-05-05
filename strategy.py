#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume spike confirmation
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR ADX < 20 (trend weakens)
# Uses 6h primary timeframe with 1d HTF for ADX and Williams %R (calculated on 1d close)
# Williams %R extremes identify exhaustion points in trending markets
# ADX filter ensures we only trade in trending conditions, avoiding whipsaws in ranges
# Volume confirmation adds momentum validation
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
    
    # Get 1d data ONCE before loop for all indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    if len(df_1d) >= 14:
        # Calculate True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1)).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean()
        
        # Calculate Directional Movement
        dm_plus = pd.Series(df_1d['high']).diff()
        dm_minus = -pd.Series(df_1d['low']).diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus.rolling(window=14, min_periods=14).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=14, min_periods=14).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus)).replace([np.inf, -np.inf], 0)
        adx = dx.rolling(window=14, min_periods=14).mean()
        
        adx_values = adx.values
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # Calculate 1d Williams %R
    if len(df_1d) >= 14:
        highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
        lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
        williams_r = -100 * ( (highest_high - pd.Series(df_1d['close'])) / (highest_high - lowest_low) )
        williams_r = williams_r.replace([np.inf, -np.inf], np.nan)
        williams_r_values = williams_r.values
        williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_values)
    else:
        williams_r_aligned = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if (williams_r_aligned[i] < -80 and 
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (recovery from oversold) OR ADX < 20 (trend weakens)
            if williams_r_aligned[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (decline from overbought) OR ADX < 20 (trend weakens)
            if williams_r_aligned[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals