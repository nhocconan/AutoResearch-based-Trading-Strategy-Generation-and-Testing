#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d ADX trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND 1d ADX < 25 (low trend) AND 4h volume > 1.5x 20-bar average.
Short when Williams %R > -20 (overbought) AND 1d ADX < 25 (low trend) AND 4h volume > 1.5x 20-bar average.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Uses 1d for ADX trend regime and 4h for execution and volume confirmation.
Designed to capture mean-reversion bounces in low-volatility, ranging markets. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX on 1d timeframe
    # True Range
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    tr14 = wilders_smoothing(tr1, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate Williams %R on 4h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) != 0, williams_r, -50)
    
    # Calculate 4h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: ADX < 25 indicates low trend (ranging market favorable for mean reversion)
        low_trend = adx_aligned[i] < 25
        
        # Mean reversion conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Exit conditions: Williams %R crosses above/below -50
        exit_long = williams_r[i] > -50
        exit_short = williams_r[i] < -50
        
        if position == 0:
            # Long: oversold with volume confirmation and low trend
            if (oversold and volume_confirmed and low_trend):
                signals[i] = 0.25
                position = 1
            # Short: overbought with volume confirmation and low trend
            elif (overbought and volume_confirmed and low_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_ADX_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0