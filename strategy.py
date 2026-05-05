#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below Camarilla S3 AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-25 trades/year per symbol.
# Camarilla provides structure; ADX filters trending markets; volume confirms breakout strength.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# 12h timeframe reduces trade frequency to minimize fee drag while capturing medium-term trends.

name = "12h_Camarilla_R3S3_1dADX_Volume_Confirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R3, S3) based on previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_high = close_12h + 1.1 * (high_12h - low_12h) / 2
    camarilla_low = close_12h - 1.1 * (high_12h - low_12h) / 2
    
    # Shift to use previous bar's levels (breakout of previous bar's Camarilla)
    camarilla_high = np.roll(camarilla_high, 1)
    camarilla_low = np.roll(camarilla_low, 1)
    camarilla_high[0] = np.nan  # First value invalid after roll
    camarilla_low[0] = np.nan
    
    # Align Camarilla levels to prices timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Uptrend when ADX > 25 and +DI > -DI, downtrend when ADX > 25 and -DI > +DI
    uptrend_1d = (adx > 25) & (di_plus > di_minus)
    downtrend_1d = (adx > 25) & (di_minus > di_plus)
    
    # Align 1d trend to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Camarilla R3 AND 1d uptrend AND volume confirmation
            if (close[i] > camarilla_high_aligned[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Camarilla S3 AND 1d downtrend AND volume confirmation
            elif (close[i] < camarilla_low_aligned[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Camarilla S3 OR 1d trend changes to downtrend
            if (close[i] < camarilla_low_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Camarilla R3 OR 1d trend changes to uptrend
            if (close[i] > camarilla_high_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals