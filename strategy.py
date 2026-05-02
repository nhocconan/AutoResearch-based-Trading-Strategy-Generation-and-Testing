#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX Trend Filter + Volume Confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings below -90 (oversold) or above -10 (overbought)
# with volume spike indicate potential reversals. 1d ADX > 25 ensures trades align with strong trend direction
# to avoid false signals in ranging markets. Designed for 50-150 total trades over 4 years (12-37/year)
# on 6h timeframe. Works in bull markets (buying oversold in uptrend) and bear markets
# (selling overbought in downtrend) by only taking trades in direction of 1d ADX trend.

name = "6h_WilliamsR_Extreme_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for trend filter (strong trend: ADX > 25)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
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
    
    # Smoothed TR, DM+,
    tr_period = 14
    tr_smooth = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    tr_smooth[tr_period-1] = tr[:tr_period].sum()
    dm_plus_smooth[tr_period-1] = dm_plus[:tr_period].sum()
    dm_minus_smooth[tr_period-1] = dm_minus[:tr_period].sum()
    
    for i in range(tr_period, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / tr_period) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.zeros_like(dx)
    adx[2*tr_period-1] = dx[tr_period:2*tr_period].mean()
    
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 2*tr_period, 30)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -90 (extreme oversold) with volume spike AND ADX > 25 (strong trend)
            if (williams_r[i] < -90 and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -10 (extreme overbought) with volume spike AND ADX > 25 (strong trend)
            elif (williams_r[i] > -10 and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -50 (recovery from oversold) OR ADX < 20 (trend weakening)
            if williams_r[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (decline from overbought) OR ADX < 20 (trend weakening)
            if williams_r[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals