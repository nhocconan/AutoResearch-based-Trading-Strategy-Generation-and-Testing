#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets (ADX < 25),
# extreme readings (%R < -80 for long, %R > -20 for short) precede mean reversion.
# Volume spike (1.5x 20-period average) confirms participation. Discrete sizing 0.25
# targets ~75-150 trades over 4 years (19-38/year) to minimize fee drag.
# Works in both bull/bear markets by adapting to regime: mean revert in range, pause in trend.

name = "6h_WilliamsR_MeanReversion_1dADX25_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ADX(14) on 1d for regime filter
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    tr_smooth = atr * 14  # Approximation for smoothed TR
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_values = np.where(np.isnan(adx_values), 0, adx_values)  # Fill NaN with 0 (low trend)
    
    # Align indicators to 6h (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation (1.5x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R and ADX calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Williams %R extreme readings for mean reversion
            williams_oversold = williams_r_aligned[i] < -80
            williams_overbought = williams_r_aligned[i] > -20
            
            # ADX regime filter: only mean revert in ranging markets (ADX < 25)
            ranging_market = adx_aligned[i] < 25
            
            if williams_oversold and ranging_market and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif williams_overbought and ranging_market and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to neutral (-50) or ADX strengthens (trend emerging)
            if williams_r_aligned[i] > -50 or adx_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral (-50) or ADX strengthens (trend emerging)
            if williams_r_aligned[i] < -50 or adx_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals