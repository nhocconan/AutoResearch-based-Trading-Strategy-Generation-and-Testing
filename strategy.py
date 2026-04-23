#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
Long when Williams %R < -80 (oversold) in low ADX regime (<25) with volume spike.
Short when Williams %R > -20 (overbought) in low ADX regime (<25) with volume spike.
Uses 12h timeframe for intermediate-term mean reversion in ranging markets.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Uses discrete position sizing (0.25) to balance return and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams %R (14-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 1d ADX (14-period) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1).subtract(pd.Series(low_1d)).abs().values
    tr2 = pd.Series(high_1d).shift(1).subtract(pd.Series(close_1d)).abs().values
    tr3 = pd.Series(low_1d).shift(1).subtract(pd.Series(close_1d)).abs().values
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((pd.Series(high_1d).diff().values > pd.Series(low_1d).shift(1).diff().values) & 
                       (pd.Series(high_1d).diff().values > 0), 
                       pd.Series(high_1d).diff().values, 0)
    dm_minus = np.where((pd.Series(low_1d).shift(1).diff().values > pd.Series(high_1d).diff().values) & 
                        (pd.Series(low_1d).shift(1).diff().values > 0), 
                        pd.Series(low_1d).shift(1).diff().values, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * (dm_plus_14 / tr_14)
    di_minus = 100 * (dm_minus_14 / tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume spike: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need Williams %R and ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: low ADX (<25) indicates ranging market
        low_adx_regime = adx_aligned[i] < 25
        
        if position == 0:
            # Long: Williams %R oversold (< -80) in ranging market with volume spike
            if williams_r_aligned[i] < -80 and low_adx_regime and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) in ranging market with volume spike
            elif williams_r_aligned[i] > -20 and low_adx_regime and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50 to -50) or ADX increases (trend developing)
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R >= -50 or ADX >= 30 (trend developing)
                if williams_r_aligned[i] >= -50 or adx_aligned[i] >= 30:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R <= -50 or ADX >= 30 (trend developing)
                if williams_r_aligned[i] <= -50 or adx_aligned[i] >= 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_MeanReversion_1dADXRegime_VolumeSpike"
timeframe = "12h"
leverage = 1.0