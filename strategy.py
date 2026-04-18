#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d volume confirmation and 1w ADX trend filter.
In trending markets (ADX > 25): Jaw-Teeth-Lips alignment gives directional bias.
Long: Lips > Teeth > Jaw + volume confirmation. Short: Lips < Teeth < Jaw + volume.
In ranging markets (ADX < 20): fade extremes at Bollinger Bands (2,2) on 12h.
Designed for 15-30 trades/year to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_ktf_data, align_ktf_to_ltf

def calculate_smma(data, period):
    """Smoothed Moving Average (used in Williams Alligator)."""
    smma = np.full(len(data), np.nan)
    if len(data) < period:
        return smma
    smma[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        smma[i] = (smma[i-1] * (period - 1) + data[i]) / period
    return smma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = np.full(len(tr), np.nan)
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    
    # First values (simple average)
    if len(tr) >= period:
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        # Wilder smoothing
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    di_plus = np.full(len(tr), np.nan)
    di_minus = np.full(len(tr), np.nan)
    dx = np.full(len(tr), np.nan)
    
    for i in range(period, len(tr)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX
    adx = np.full(len(tr), np.nan)
    for i in range(2*period, len(tr)):
        if not np.isnan(dx[i-1]):
            if i == 2*period:
                adx[i] = np.nanmean(dx[period:2*period+1])
            else:
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_bbands(close, period=20, std_dev=2):
    """Bollinger Bands."""
    if len(close) < period:
        return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)
    
    sma = np.full(len(close), np.nan)
    std = np.full(len(close), np.nan)
    
    for i in range(period-1, len(close)):
        sma[i] = np.mean(close[i-period+1:i+1])
        std[i] = np.std(close[i-period+1:i+1])
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_ktf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_ktf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams Alligator on 12h (using SMMA)
    jaw_period, teeth_period, lips_period = 13, 8, 5
    jaw_offset, teeth_offset, lips_offset = 8, 5, 3
    
    jaw = calculate_smma(close, jaw_period)
    teeth = calculate_smma(close, teeth_period)
    lips = calculate_smma(close, lips_period)
    
    # Shift to avoid look-ahead (Alligator uses future values)
    jaw = np.roll(jaw, jaw_offset)
    teeth = np.roll(teeth, teeth_offset)
    lips = np.roll(lips, lips_offset)
    # Set NaN for rolled values
    jaw[:jaw_offset] = np.nan
    teeth[:teeth_offset] = np.nan
    lips[:lips_offset] = np.nan
    
    # ADX on 1w
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Bollinger Bands on 12h for ranging market signals
    bb_upper, bb_middle, bb_lower = calculate_bbands(close, 20, 2)
    
    # Volume moving average (20-period) on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align all to 12h timeframe
    jaw_12h = align_ktf_to_ltf(prices, df_1d, jaw)  # Using 1d as bridge for alignment
    teeth_12h = align_ktf_to_ltf(prices, df_1d, teeth)
    lips_12h = align_ktf_to_ltf(prices, df_1d, lips)
    adx_12h = align_ktf_to_ltf(prices, df_1w, adx_1w)
    bb_upper_12h = align_ktf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_12h = align_ktf_to_ltf(prices, df_1d, bb_lower)
    vol_ma_12h = align_ktf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period, 20) + max(jaw_offset, teeth_offset, lips_offset)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(bb_upper_12h[i]) or np.isnan(bb_lower_12h[i]) or
            np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma_12h[i]
        
        if position == 0:
            # Trending market (ADX > 25): Alligator alignment
            if adx_12h[i] > 25:
                # Long: Lips > Teeth > Jaw (bullish alignment)
                if lips_12h[i] > teeth_12h[i] > jaw_12h[i] and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short: Lips < Teeth < Jaw (bearish alignment)
                elif lips_12h[i] < teeth_12h[i] < jaw_12h[i] and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (ADX < 20): Bollinger Band fade
            elif adx_12h[i] < 20:
                # Long: price at or below lower BB
                if close[i] <= bb_lower_12h[i] and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short: price at or above upper BB
                elif close[i] >= bb_upper_12h[i] and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if adx_12h[i] > 25:
                # Exit trend: Alligator alignment breaks
                if not (lips_12h[i] > teeth_12h[i] > jaw_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit range: price crosses above middle BB
                if close[i] >= bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if adx_12h[i] > 25:
                # Exit trend: Alligator alignment breaks
                if not (lips_12h[i] < teeth_12h[i] < jaw_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit range: price crosses below middle BB
                if close[i] <= bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_1wADX"
timeframe = "12h"
leverage = 1.0