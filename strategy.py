#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Williams %R with volume confirmation and ADX trend filter.
# Williams %R identifies overbought/oversold conditions on daily timeframe.
# Enter long when %R crosses above -80 from below (oversold bounce) with volume confirmation.
# Enter short when %R crosses below -20 from above (overbought rejection) with volume confirmation.
# Filter trades by ADX(14) > 25 on 1-day timeframe to ensure trending markets.
# Designed to capture mean reversion in trends, working in both bull and bear markets.
# Target: 20-30 trades/year per signal with disciplined entries.
name = "6h_WilliamsR_ADX_Volume_Trend"
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
    
    # Daily Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Daily ADX (14-period) for trend strength
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).values - pd.Series(df_1d['low']).values
    tr2 = np.abs(pd.Series(df_1d['high']).values - pd.Series(df_1d['close']).shift(1).values)
    tr3 = np.abs(pd.Series(df_1d['low']).values - pd.Series(df_1d['close']).shift(1).values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((pd.Series(df_1d['high']).values - pd.Series(df_1d['high']).shift(1).values) > 
                       (pd.Series(df_1d['low']).shift(1).values - pd.Series(df_1d['low']).values),
                       np.maximum(pd.Series(df_1d['high']).values - pd.Series(df_1d['high']).shift(1).values, 0), 0)
    dm_minus = np.where((pd.Series(df_1d['low']).shift(1).values - pd.Series(df_1d['low']).values) > 
                        (pd.Series(df_1d['high']).values - pd.Series(df_1d['high']).shift(1).values),
                        np.maximum(pd.Series(df_1d['low']).shift(1).values - pd.Series(df_1d['low']).values, 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smoothed = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smoothed = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    tr_smoothed = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below, ADX > 25, volume spike
            if (williams_r_aligned[i] > -80 and 
                williams_r_aligned[i-1] <= -80 and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above, ADX > 25, volume spike
            elif (williams_r_aligned[i] < -20 and 
                  williams_r_aligned[i-1] >= -20 and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Williams %R rises above -20 (overbought) or ADX weakens
            if (williams_r_aligned[i] >= -20) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Williams %R falls below -80 (oversold) or ADX weakens
            if (williams_r_aligned[i] <= -80) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals