#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Uses Williams Alligator (Jaw=TEETH=13, Teeth=TEETH=8, Lips=TEETH=5) from 6h for trend direction and alignment
# 1d EMA34 for higher timeframe trend filter to avoid counter-trend whipsaws
# Volume spike (>2.0x 50-bar average) confirms breakout strength
# Discrete sizing 0.25 to balance return and drawdown; target 80-120 total trades over 4 years (20-30/year)
# Williams Alligator provides smoothed trend identification with built-in filtering, effective in both bull and bear markets

name = "6h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_6h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 6h
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(source, length):
        if length < 1:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        for i in range(len(source)):
            if i < length:
                result[i] = np.nan
            elif i == length:
                result[i] = np.nanmean(source[:i+1])
            else:
                result[i] = (result[i-1] * (length - 1) + source[i]) / length
        return result
    
    jaw = smma(close_6h, 13)
    teeth = smma(close_6h, 8)
    lips = smma(close_6h, 5)
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume filter (>2.0x 50-bar average)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (2.0 * vol_ma_50)
    
    # Align HTF indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_filter_aligned = align_htf_to_ltf(prices, df_6h, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND above 1d EMA34 AND volume spike
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and \
               close[i] > ema34_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND below 1d EMA34 AND volume spike
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and \
                 close[i] < ema34_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines cross or price below 1d EMA34
            if lips_aligned[i] < teeth_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines cross or price above 1d EMA34
            if lips_aligned[i] > teeth_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals