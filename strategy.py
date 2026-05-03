#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter + volume spike confirmation.
# Long when Williams %R(14) crosses above -80 (oversold bounce) in 1d uptrend (ADX>25 and +DI>-DI) with volume spike.
# Short when Williams %R(14) crosses below -20 (overbought rejection) in 1d downtrend (ADX>25 and +DI<+DI) with volume spike.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 50-150 total trades over 4 years.
# Williams %R identifies overextended moves, 1d ADX ensures higher timeframe trend alignment,
# Volume spike confirms institutional interest. Works in both bull and bear markets by only trading with
# the 1d trend, avoiding counter-trend whipsaws. Designed for 6h timeframe to minimize fee drag.

name = "6h_WilliamsR_1dADX_VolumeSpike"
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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate 6h Williams %R
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams %R = (highest high - close) / (highest high - lowest low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI values
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align ADX, DI+ and DI- to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus)
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(di_plus_aligned[i]) or np.isnan(di_minus_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        wr = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        di_plus_val = di_plus_aligned[i]
        di_minus_val = di_minus_aligned[i]
        vol_spike = volume_spike[i]
        
        # Trend conditions: ADX > 25 indicates strong trend
        strong_uptrend = adx_val > 25 and di_plus_val > di_minus_val
        strong_downtrend = adx_val > 25 and di_plus_val < di_minus_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) in uptrend with volume spike
            if wr > -80 and wr <= -75 and strong_uptrend and vol_spike:  # Crossing above -80
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought rejection) in downtrend with volume spike
            elif wr < -20 and wr >= -15 and strong_downtrend and vol_spike:  # Crossing below -20
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -20 (overbought) or trend weakens
            if wr >= -20 or adx_val < 20:  # Overbought or trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -80 (oversold) or trend weakens
            if wr <= -80 or adx_val < 20:  # Oversold or trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals