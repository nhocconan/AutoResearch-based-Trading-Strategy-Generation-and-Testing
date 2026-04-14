#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (-20/-80 thresholds)
# ADX > 25 filters for trending markets where mean reversion works best
# Volume > 1.5x average confirms institutional participation in the reversal
# Works in bull/bear as ADX adapts to trend strength and Williams %R captures reversals
# Target: 20-30 trades/year per symbol (80-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX (14) on 1d data
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = df_1d['high'] - df_1d['high'].shift(1)
    dm_minus = df_1d['low'].shift(1) - df_1d['low']
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    atr = tr.rolling(window=14, min_periods=14).mean()
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).sum()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 6h timeframe
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Williams %R (14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    wr_values = wr.values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 14, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(wr_values[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX trend filter: trending market (ADX > 25)
        trending = adx_aligned[i] > 25
        
        # Williams %R signals
        oversold = wr_values[i] < -80
        overbought = wr_values[i] > -20
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: oversold + trending + volume
            if oversold and trending and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: overbought + trending + volume
            elif overbought and trending and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) or overbought
            if wr_values[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) or oversold
            if wr_values[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_WilliamsR_ADX_Volume_v1"
timeframe = "6h"
leverage = 1.0