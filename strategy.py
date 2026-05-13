#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w ADX(14) regime filter and volume confirmation.
# Long when price breaks above Donchian upper band with 1w ADX > 20 (trending) and volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower band with 1w ADX > 20 and volume > 1.5x average.
# Exit when price reverses and closes below/above the midpoint of the Donchian channel.
# Uses discrete position sizing 0.25. Target: 30-100 total trades over 4 years on 1d timeframe.
# ADX regime filter ensures we only trade in strong trends, avoiding whipsaws in ranging markets.
# Volume confirmation validates breakout strength. Donchian exit provides clear, objective stop.

name = "1d_Donchian20_1wADX_Regime_Volume_Breakout_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel (20-period)
    lookback = 20
    if n < lookback + 1:
        return np.zeros(n)
    
    upper_band = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower_band = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    mid_band = (upper_band + lower_band) / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 1w data
    period = 14
    if len(close_1w) < period + 1:
        adx_1w = np.full(len(close_1w), np.nan)
    else:
        # True Range
        tr1 = pd.Series(high_1w).diff().abs()
        tr2 = (pd.Series(high_1w) - pd.Series(close_1w).shift(1)).abs()
        tr3 = (pd.Series(low_1w) - pd.Series(close_1w).shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
        
        # Directional Movement
        up_move = pd.Series(high_1w).diff()
        down_move = -pd.Series(low_1w).diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR, +DM, -DM
        tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_1w = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    
    # Align 1w ADX to 1d timeframe (wait for 1w bar to close)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper band with ADX > 20 and volume spike
            if (close[i] > upper_band[i] and 
                adx_1w_aligned[i] > 20 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower band with ADX > 20 and volume spike
            elif (close[i] < lower_band[i] and 
                  adx_1w_aligned[i] > 20 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below mid-band (reversal signal)
            if close[i] < mid_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above mid-band (reversal signal)
            if close[i] > mid_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals