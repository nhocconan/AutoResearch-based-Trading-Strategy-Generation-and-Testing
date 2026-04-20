#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Williams %R + 1-week ADX trend filter
# Williams %R identifies overbought/oversold conditions on daily timeframe
# Weekly ADX > 25 indicates strong trend (use only in trending markets)
# In trending markets (ADX > 25): follow Williams %R momentum (buy oversold in uptrend, sell overbought in downtrend)
# In ranging markets (ADX <= 25): mean revert at extreme Williams %R levels
# Designed to work in both bull and bear markets by adapting to trend strength
# Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R on daily timeframe
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    wr_1d = np.where((highest_high - lowest_low) == 0, -50, wr)  # Handle division by zero
    wr_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    
    # Load weekly data for ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX on weekly timeframe
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1w = np.where((di_plus + di_minus) == 0, 0, adx)  # Handle division by zero
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(wr_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr_val = wr_1d_aligned[i]
        adx_val = adx_1w_aligned[i]
        
        # Regime classification based on ADX
        is_trending = adx_val > 25  # Strong trend
        is_ranging = adx_val <= 25  # Weak trend/ranging
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Enter long conditions
            long_signal = False
            if is_trending:
                # In trending markets: buy when oversold (Williams %R > -80 from below)
                if wr_val > -80 and wr_1d_aligned[i-1] <= -80:
                    long_signal = True
            elif is_ranging:
                # In ranging markets: mean revert at extreme oversold
                if wr_val > -90:  # Recovering from deep oversold
                    long_signal = True
            
            # Enter short conditions
            short_signal = False
            if is_trending:
                # In trending markets: sell when overbought (Williams %R < -20 from above)
                if wr_val < -20 and wr_1d_aligned[i-1] >= -20:
                    short_signal = True
            elif is_ranging:
                # In ranging markets: mean revert at extreme overbought
                if wr_val < -10:  # Declining from extreme overbought
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: overbought condition or trend weakening
            exit_signal = False
            if wr_val < -20:  # Williams %R overbought
                exit_signal = True
            elif is_trending and adx_val < 20:  # Trend weakening
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: oversold condition or trend weakening
            exit_signal = False
            if wr_val > -80:  # Williams %R oversold
                exit_signal = True
            elif is_trending and adx_val < 20:  # Trend weakening
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_ADX_TrendFilter"
timeframe = "1d"
leverage = 1.0