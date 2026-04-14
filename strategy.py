#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R with volume confirmation and ADX trend filter.
# Long when Williams %R crosses above -80 (oversold reversal) on 1d timeframe, ADX > 20 (trending), and volume > 1.3x average.
# Short when Williams %R crosses below -20 (overbought reversal) on 1d timeframe, ADX > 20, and volume > 1.3x average.
# Exit when Williams %R returns to -50 (mean reversion) or ADX drops below 15 (trend weakening).
# Williams %R identifies overbought/oversold conditions, ADX confirms trend strength, volume validates participation.
# Designed to work in both bull and bear markets by trading mean reversions within trends.
# Target: 25-40 trades/year per symbol (100-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for Williams %R(14) and ADX(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ADX (14)
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
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1.3x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # Need Williams %R and ADX periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        # Weak trend filter: ADX < 15 indicates trend weakening
        weak_trend = adx_aligned[i] < 15
        
        if position == 0:
            # Look for Williams %R reversals in trending market
            # Long: Williams %R crosses above -80 (oversold reversal) AND trending AND volume confirmation
            if i > start and (williams_r_aligned[i-1] <= -80 and williams_r_aligned[i] > -80 and 
                trending and volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: Williams %R crosses below -20 (overbought reversal) AND trending AND volume confirmation
            elif i > start and (williams_r_aligned[i-1] >= -20 and williams_r_aligned[i] < -20 and 
                  trending and volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to -50 or trend weakens
            if (williams_r_aligned[i] >= -50 or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to -50 or trend weakens
            if (williams_r_aligned[i] <= -50 or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_ADX_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0