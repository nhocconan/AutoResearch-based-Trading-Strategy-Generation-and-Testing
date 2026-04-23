#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years). Uses discrete position sizing (0.25) to minimize fee churn.
Williams %R identifies overbought/oversold conditions, ADX filters for ranging markets (ADX < 25) where mean reversion works best,
and volume confirmation avoids low-liquidity false signals. Works in both bull/bear via ADX regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams %R (14-period) for mean reversion signals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low + 1e-10) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 1d ADX (14-period) for regime filter - ranging markets (ADX < 25) favor mean reversion
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need Williams %R, ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R signals: Long when oversold (< -80), Short when overbought (> -20)
        williams_long_signal = williams_r_aligned[i] < -80
        williams_short_signal = williams_r_aligned[i] > -20
        
        # Regime filter: Only trade in ranging markets (ADX < 25) where mean reversion works
        ranging_market = adx_aligned[i] < 25
        
        # Volume filter: 12h volume > 1.5x 20-period MA (moderate to avoid excessive trades)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Oversold + ranging market + volume confirmation
            if williams_long_signal and ranging_market and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Overbought + ranging market + volume confirmation
            elif williams_short_signal and ranging_market and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50 to -50) or ADX trends up
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R rises above -50 (returns from oversold)
                if williams_r_aligned[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R falls below -50 (returns from overbought)
                if williams_r_aligned[i] < -50:
                    exit_signal = True
            
            # Also exit if market starts trending (ADX >= 30) - unfavorable for mean reversion
            if adx_aligned[i] >= 30:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_MeanReversion_1dADX_Regime_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0