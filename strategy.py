#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + 1-day Williams %R reversal strategy
# ADX(14) > 25 indicates trending market, < 20 indicates ranging market
# In trending markets: Williams %R < -80 = long setup, > -20 = short setup
# In ranging markets: Williams %R > -50 = long setup (buy dip), < -50 = short setup (sell rally)
# Volume confirmation: require volume > 1.3x 20-period average
# Designed to adapt to market regime using ADX and Williams %R
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Williams %R and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R on daily timeframe
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    wr_1d = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d + 1e-10)
    wr_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    
    # Calculate 6h ADX for regime detection
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(adx[i]) or np.isnan(wr_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        wr = wr_1d_aligned[i]
        
        if position == 0:
            # Entry logic based on regime
            long_signal = False
            short_signal = False
            
            if has_volume:
                if is_trending:
                    # Trending: fade extreme Williams %R
                    if wr < -80:  # Oversold = long
                        long_signal = True
                    elif wr > -20:  # Overbought = short
                        short_signal = True
                elif is_ranging:
                    # Ranging: fade mid-range Williams %R
                    if wr > -50:  # Above midpoint = long (buy dip)
                        long_signal = True
                    elif wr < -50:  # Below midpoint = short (sell rally)
                        short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: opposite condition based on regime
            exit_signal = False
            if has_volume:
                if is_trending and wr > -20:  # Overbought in trend = exit
                    exit_signal = True
                elif is_ranging and wr < -50:  # Below midpoint in range = exit
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: opposite condition based on regime
            exit_signal = False
            if has_volume:
                if is_trending and wr < -80:  # Oversold in trend = exit
                    exit_signal = True
                elif is_ranging and wr > -50:  # Above midpoint in range = exit
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsR_RegimeAdaptive"
timeframe = "6h"
leverage = 1.0