#!/usr/bin/env python3
"""
Hypothesis: 4H Williams Alligator crossover with 1D volume confirmation and ADX trend filter.
Williams Alligator (Jaw/Teeth/Lips) identifies trend phases - crossover signals new trends.
Volume confirms conviction, ADX filters for trending vs ranging markets.
Designed for 15-25 trades/year to minimize fee drag while capturing strong trends.
Works in bull markets (buy golden cross) and bear markets (buy death cross for shorts).
"""

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
    
    # Williams Alligator on 4H: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Calculate SMAs for Alligator lines
    jaw = np.full(n, np.nan)
    teeth = np.full(n, np.nan)
    lips = np.full(n, np.nan)
    
    for i in range(max(jaw_period, teeth_period, lips_period) - 1, n):
        if i >= jaw_period - 1:
            jaw[i] = np.mean(close[i - jaw_period + 1:i + 1])
        if i >= teeth_period - 1:
            teeth[i] = np.mean(close[i - teeth_period + 1:i + 1])
        if i >= lips_period - 1:
            lips[i] = np.mean(close[i - lips_period + 1:i + 1])
    
    # Shift the lines forward (Alligator-specific)
    jaw_shifted = np.full(n, np.nan)
    teeth_shifted = np.full(n, np.nan)
    lips_shifted = np.full(n, np.nan)
    
    for i in range(jaw_shift, n):
        jaw_shifted[i] = jaw[i - jaw_shift]
    for i in range(teeth_shift, n):
        teeth_shifted[i] = teeth[i - teeth_shift]
    for i in range(lips_shift, n):
        lips_shifted[i] = lips[i - lips_shift]
    
    # Get 1D data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1D volume moving average (20-period)
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align 1D volume MA to 4H timeframe
    vol_ma_1d_4h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1D data for ADX trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1D
    adx_period = 14
    tr = np.zeros(len(close_1d))
    plus_dm = np.zeros(len(close_1d))
    minus_dm = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        high_low = high_1d[i] - low_1d[i]
        high_prev_close = abs(high_1d[i] - close_1d[i-1])
        low_prev_close = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(high_low, high_prev_close, low_prev_close)
        
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        
        # Adjust for inside/outside bars
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
    
    # Calculate smoothed values
    atr = np.zeros(len(close_1d))
    smoothed_plus_dm = np.zeros(len(close_1d))
    smoothed_minus_dm = np.zeros(len(close_1d))
    
    # Initial values
    if len(close_1d) >= adx_period:
        atr[adx_period-1] = np.mean(tr[:adx_period])
        smoothed_plus_dm[adx_period-1] = np.mean(plus_dm[:adx_period])
        smoothed_minus_dm[adx_period-1] = np.mean(minus_dm[:adx_period])
        
        # Wilder smoothing
        for i in range(adx_period, len(close_1d)):
            atr[i] = (atr[i-1] * (adx_period-1) + tr[i]) / adx_period
            smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * (adx_period-1) + plus_dm[i]) / adx_period
            smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * (adx_period-1) + minus_dm[i]) / adx_period
    
    # Calculate DI and DX
    plus_di = np.zeros(len(close_1d))
    minus_di = np.zeros(len(close_1d))
    dx = np.zeros(len(close_1d))
    
    for i in range(adx_period-1, len(close_1d)):
        if atr[i] != 0:
            plus_di[i] = 100 * smoothed_plus_dm[i] / atr[i]
            minus_di[i] = 100 * smoothed_minus_dm[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros(len(close_1d))
    if len(close_1d) >= 2*adx_period-1:
        adx[2*adx_period-2] = np.mean(dx[adx_period-1:2*adx_period-1])
        for i in range(2*adx_period-1, len(close_1d)):
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align 1D ADX to 4H timeframe
    adx_1d_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(
        jaw_shifted.shape[0] if not np.all(np.isnan(jaw_shifted)) else 0,
        teeth_shifted.shape[0] if not np.all(np.isnan(teeth_shifted)) else 0,
        lips_shifted.shape[0] if not np.all(np.isnan(lips_shifted)) else 0,
        vol_ma_1d_4h.shape[0] if not np.all(np.isnan(vol_ma_1d_4h)) else 0,
        adx_1d_4h.shape[0] if not np.all(np.isnan(adx_1d_4h)) else 0
    )
    
    # Ensure we have enough data
    start_idx = max(start_idx, 30)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(vol_ma_1d_4h[i]) or 
            np.isnan(adx_1d_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4H volume > 1.5 * 1D average volume
        vol_confirmed = volume[i] > 1.5 * vol_ma_1d_4h[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_4h[i] > 25
        
        # Alligator crossovers
        lips_above_teeth = lips_shifted[i] > teeth_shifted[i]
        teeth_above_jaw = teeth_shifted[i] > jaw_shifted[i]
        lips_below_teeth = lips_shifted[i] < teeth_shifted[i]
        teeth_below_jaw = teeth_shifted[i] < jaw_shifted[i]
        
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            # Long entry: bullish alignment with volume and trend
            if bullish_alignment and vol_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment with volume and trend
            elif bearish_alignment and vol_confirmed and trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: bearish alignment or loss of trend/volume
            if bearish_alignment or not trending or not vol_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish alignment or loss of trend/volume
            if bullish_alignment or not trending or not vol_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0