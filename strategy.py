#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Williams %R (14-period) for overbought/oversold conditions,
# combined with volume confirmation and ADX trend strength filter. Enters long when Williams %R < -80
# (oversold) with volume spike and ADX > 25 (trending), exits when Williams %R > -20 (overbought).
# Shorts when Williams %R > -20 (overbought) with volume spike and ADX > 25, exits when Williams %R < -80.
# Designed to capture mean reversion in trending markets, avoiding chop. Target: 15-25 trades/year.
name = "12h_WilliamsR_ADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1-day ADX (14-period) for trend strength
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).rolling(window=2).max().values - pd.Series(df_1d['low']).rolling(window=2).min().values
    tr2 = abs(pd.Series(df_1d['high']).rolling(window=2).max().values - pd.Series(df_1d['close']).shift(1).values)
    tr3 = abs(pd.Series(df_1d['low']).rolling(window=2).min().values - pd.Series(df_1d['close']).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Handle first value where shift creates NaN
    tr[0] = df_1d['high'][0] - df_1d['low'][0]
    
    # Directional Movement
    dm_plus = np.where((pd.Series(df_1d['high']).diff().values > 0) & (pd.Series(df_1d['high']).diff().values > -pd.Series(df_1d['low']).diff().values),
                       pd.Series(df_1d['high']).diff().values, 0)
    dm_minus = np.where((-pd.Series(df_1d['low']).diff().values > 0) & (-pd.Series(df_1d['low']).diff().values > pd.Series(df_1d['high']).diff().values),
                        -pd.Series(df_1d['low']).diff().values, 0)
    # Handle first value where diff creates NaN
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    atr[0] = tr[0]
    dm_plus_smooth[0] = dm_plus[0]
    dm_minus_smooth[0] = dm_minus[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Avoid division by zero
    dm_plus_smooth = np.where(atr == 0, 0, dm_plus_smooth)
    dm_minus_smooth = np.where(atr == 0, 0, dm_minus_smooth)
    
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * abs(di_plus - di_minus) / (di_plus + di_minus))
    
    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    if len(dx) >= 14:
        adx[13] = np.mean(dx[0:14])
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), volume spike, ADX > 25 (trending)
            if (williams_r_aligned[i] < -80 and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), volume spike, ADX > 25 (trending)
            elif (williams_r_aligned[i] > -20 and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Williams %R > -20 (overbought)
            if williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Williams %R < -80 (oversold)
            if williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals