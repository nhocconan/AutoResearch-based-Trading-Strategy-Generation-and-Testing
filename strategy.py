#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX regime filter and volume confirmation
# Williams %R(14) identifies overbought/oversold conditions for mean reversion entries
# 1d ADX > 25 filters for trending markets where reversals are more reliable
# Volume spike confirms institutional participation at reversal points
# Works in both bull and bear markets by using ADX regime to avoid choppy false signals
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Discrete position sizing: 0.25 (25% of capital) balances opportunity and risk

name = "6h_WilliamsR_Reversal_1dADX_Volume"
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
    
    # Calculate 1d Williams %R(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Replace division by zero with -50 (neutral)
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    
    # Calculate 1d ADX(14) for regime filter
    # ADX requires +DI and -DI calculation
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Replace division by zero with 0
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) with volume spike AND ADX > 25 (trending market)
            if (williams_r_1d_aligned[i] < -80 and 
                volume_spike[i] and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) with volume spike AND ADX > 25 (trending market)
            elif (williams_r_1d_aligned[i] > -20 and 
                  volume_spike[i] and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns above -50 (mean reversion complete) OR ADX < 20 (choppy market)
            if williams_r_1d_aligned[i] > -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (mean reversion complete) OR ADX < 20 (choppy market)
            if williams_r_1d_aligned[i] < -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals