#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and chop regime filter
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) smoothed with SMMA for trend identification
# Volume confirmation: 12h volume > 1.5x 20-period EMA to filter weak signals
# Chop regime: Choppiness Index(14) > 61.8 for ranging markets (mean reversion at Alligator lines)
# Designed for 12h timeframe targeting 12-37 trades/year with discrete sizing (0.25)
# Works in bull markets (buy when price > Lips in uptrend Alligator alignment)
# Works in bear markets (sell when price < Lips in downtrend Alligator alignment)
# Williams Alligator provides smooth trend identification with built-in filtering
# 1d timeframe aligns with HTF reference and reduces noise vs lower timeframes

name = "12h_WilliamsAlligator_1dVolumeChop_Regime"
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
    
    # Get 1d data for chop regime and volume EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14) and sum of true ranges
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(tr_sum / (atr_14 * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop = np.where(atr_14 > 0, chop, 50.0)  # Default to 50 when ATR is 0
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 1d data for volume EMA
    vol_1d = df_1d['volume'].values
    vol_ema_20 = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator using SMMA (Smoothed Moving Average)
    # SMMA today = (SMMA yesterday * (period-1) + price today) / period
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    close_12h = df_12h['close'].values
    jaw = smma(close_12h, 13)  # Jaw (Blue) - 13-period SMMA
    teeth = smma(close_12h, 8)  # Teeth (Red) - 8-period SMMA
    lips = smma(close_12h, 5)   # Lips (Green) - 5-period SMMA
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        # Chop regime: > 61.8 = ranging market (mean reversion)
        is_ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price > Lips + volume confirmation + ranging market
            if (close[i] > lips_aligned[i] and volume_confirmed and is_ranging):
                signals[i] = 0.25
                position = 1
            # Short: price < Lips + volume confirmation + ranging market
            elif (close[i] < lips_aligned[i] and volume_confirmed and is_ranging):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Teeth (mean reversion in ranging market) OR chop < 38.2 (trending)
            if close[i] < teeth_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Teeth (mean reversion in ranging market) OR chop < 38.2 (trending)
            if close[i] > teeth_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals