#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX trend filter and Bollinger Band squeeze
# Williams %R identifies overbought/oversold conditions in 6h timeframe
# ADX(14) from 1d timeframe filters for trending vs ranging markets
# Bollinger Band width percentile identifies low volatility squeeze conditions
# Long: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND BB width < 20th percentile (squeeze)
# Short: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND BB width < 20th percentile (squeeze)
# Exit: Williams %R crosses above -50 for long, below -50 for short
# Uses mean reversion in squeeze conditions with trend filter to avoid false signals
# Timeframe: 6h, HTF: 1d for ADX and Bollinger Bands. Target: 80-180 total trades over 4 years (20-45/year).

name = "6h_WilliamsR_ADX_BBSqueeze"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for ADX and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for indicators
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_period = 14
    atr = WilderSmoothing(tr, atr_period)
    dm_plus_smooth = WilderSmoothing(dm_plus, atr_period)
    dm_minus_smooth = WilderSmoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmoothing(dx, atr_period)
    adx_1d = adx  # already aligned to 1d index
    
    # Calculate 1d Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + bb_std * std_20
    bb_lower = sma_20 - bb_std * std_20
    bb_width = bb_upper - bb_lower
    
    # Calculate Bollinger Band width percentile (using 252-day lookback ~1 year)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    lookback = 252
    for i in range(lookback, len(bb_width)):
        if i >= lookback:
            historical_width = bb_width[i-lookback:i]
            current_width = bb_width[i]
            if not np.isnan(current_width) and len(historical_width) > 0:
                percentile = np.sum(historical_width <= current_width) / len(historical_width) * 100
                bb_width_percentile[i] = percentile
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Calculate 6h Williams %R(14)
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for Bollinger Band squeeze (width < 20th percentile)
        bb_squeeze = bb_width_percentile_aligned[i] < 20
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND BB squeeze
            if (williams_r[i] < -80 and 
                adx_1d_aligned[i] > 25 and 
                bb_squeeze):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND BB squeeze
            elif (williams_r[i] > -20 and 
                  adx_1d_aligned[i] > 25 and 
                  bb_squeeze):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (recovering from oversold)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (declining from overbought)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals