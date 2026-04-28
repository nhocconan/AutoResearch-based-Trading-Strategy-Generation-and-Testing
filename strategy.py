#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d ADX25 regime filter
# Williams %R identifies overbought/oversold conditions (-20 to 0 = overbought, -80 to -100 = oversold)
# Long when %R crosses above -80 from below (oversold bounce) AND ADX > 25 (trending regime)
# Short when %R crosses below -20 from above (overbought rejection) AND ADX > 25
# Uses 1d ADX for regime filter to avoid whipsaw in ranging markets
# Williams %R is responsive yet less noisy than RSI for reversal signals
# Target: 12-37 trades/year via regime filter reducing counter-trend trades
# Works in both bull and bear markets by only trading when ADX confirms trending conditions

name = "6h_WilliamsR_Extreme_1dADX25_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smoothed DM
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx) | (di_plus + di_minus == 0), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Prepend zeros for alignment (lost first bar in TR, then 14 for ADX smoothing)
    adx = np.concatenate([np.full(27, np.nan), adx])
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R(14) on 6h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 27)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R crosses above -80 from below (oversold bounce) AND ADX > 25
            if wr_prev <= -80 and wr > -80 and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -20 from above (overbought rejection) AND ADX > 25
            elif wr_prev >= -20 and wr < -20 and adx_val > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when %R crosses below -50 (momentum loss) or ADX < 20
            if wr < -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when %R crosses above -50 (momentum loss) or ADX < 20
            if wr > -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals