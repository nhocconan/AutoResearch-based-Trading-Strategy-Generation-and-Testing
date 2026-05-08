#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX filter and volume confirmation.
# Long when Williams %R crosses above -20 from below AND 1d ADX > 25 (trending) AND volume > 1.3x 20-period average.
# Short when Williams %R crosses below -80 from above AND 1d ADX > 25 (trending) AND volume > 1.3x 20-period average.
# Exit when Williams %R crosses back below -80 for longs or above -20 for shorts.
# This strategy captures momentum extremes in trending markets with volume confirmation.
# Williams %R identifies overbought/oversold conditions. ADX ensures we only trade in trending markets.
# Volume confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_ADX_Volume"
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
    
    # 1d data for ADX and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ and DM- (14-period)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Align indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -20 from below, ADX > 25, volume filter
            williams_cross_up = (williams_r_aligned[i] > -20) and (williams_r_aligned[i-1] <= -20)
            long_cond = williams_cross_up and (adx_aligned[i] > 25) and volume_filter[i]
            
            # Short conditions: Williams %R crosses below -80 from above, ADX > 25, volume filter
            williams_cross_down = (williams_r_aligned[i] < -80) and (williams_r_aligned[i-1] >= -80)
            short_cond = williams_cross_down and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back below -80
            williams_cross_down_exit = (williams_r_aligned[i] < -80) and (williams_r_aligned[i-1] >= -80)
            if williams_cross_down_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back above -20
            williams_cross_up_exit = (williams_r_aligned[i] > -20) and (williams_r_aligned[i-1] <= -20)
            if williams_cross_up_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals