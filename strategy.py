#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d ADX trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND 6h volume > 2.0x 20-bar average.
Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND 6h volume > 2.0x 20-bar average.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Uses 1d for ADX trend regime and 6h for execution, Williams %R, and volume confirmation.
Designed to capture mean-reversion pullbacks within strong trends on higher timeframe, which works in both bull and bear markets.
Target: 12-30 trades/year per symbol.
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.maximum(high_1d - low_1d, 
                     np.absolute(high_1d - np.roll(close_1d, 1)),
                     np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Smooth TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr1 = pd.Series(tr1).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm1 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm1 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Directional Indicators
    plus_di1 = 100 * plus_dm1 / atr1
    minus_di1 = 100 * minus_dm1 / atr1
    # DX and ADX
    dx1 = 100 * np.absolute(plus_di1 - minus_di1) / (plus_di1 + minus_di1)
    adx1 = pd.Series(dx1).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) > 0, williams_r, -50.0)
    
    # Calculate 6h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx1_aligned = align_htf_to_ltf(prices, df_1d, adx1)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx1_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x 20-bar average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx1_aligned[i] > 25
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_long = williams_r[i] > -50  # cross above -50
        exit_short = williams_r[i] < -50  # cross below -50
        
        if position == 0:
            # Long: oversold in strong trend with volume confirmation
            if (oversold and strong_trend and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: overbought in strong trend with volume confirmation
            elif (overbought and strong_trend and volume_confirmed):
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

name = "6h_WilliamsR_MeanReversion_ADXTrend_Volume"
timeframe = "6h"
leverage = 1.0