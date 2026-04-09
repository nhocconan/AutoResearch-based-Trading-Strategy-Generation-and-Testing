#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1d ADX > 25 ensures we only trade in trending markets (avoids chop)
# Volume spike (>2x 20-period average) confirms momentum behind the reversal
# Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: ADX filters trending markets, Williams %R captures reversals within trends

name = "6h_1d_williamsr_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).shift(1).diff() * -1
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 20-period average volume for volume spike confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_6h[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(atr_1d)):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2x 20-period average
        volume_spike = volume[i] > 2.0 * avg_volume[i]
        
        # ADX filter: only trade when trending (ADX > 25)
        trending = adx_1d_6h[i] > 25
        
        if position == 1:  # Long position
            # Exit: Williams %R returns above -20 (overbought) OR ADX weakens
            if williams_r[i] > -20 or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -80 (oversold) OR ADX weakens
            if williams_r[i] < -80 or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume spike and ADX filter
            if volume_spike and trending:
                # Long entry: Williams %R crosses below -80 (oversold) in uptrend
                if williams_r[i] < -80 and williams_r[i-1] >= -80:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R crosses above -20 (overbought) in downtrend
                elif williams_r[i] > -20 and williams_r[i-1] <= -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals