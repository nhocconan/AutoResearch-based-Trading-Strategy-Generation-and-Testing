#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter
# Elder Ray measures bull/bear power vs EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND ADX > 25 (trending) AND EMA13 rising
# Short when Bear Power < 0 AND ADX > 25 AND EMA13 falling
# Uses 1d EMA13 for Elder Ray calculation and 1d ADX for regime filter
# Exits when power reverses or ADX < 20 (range regime)
# Designed to work in both bull and bear markets by adapting to trend strength via ADX
# Target: 12-35 trades/year via strict trend + power confirmation

name = "6h_ElderRay_Power_1dADX25_Regime_EMA13Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Elder Ray and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close for Elder Ray
    close_1d = pd.Series(df_1d['close'])
    ema13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1d EMA13 to 6h timeframe (completed 1d candles only)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Bull Power and Bear Power on 1d
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align powers to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate ADX on 1d for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR (14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed DM and ATR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_smooth
    di_minus = 100 * dm_minus_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient history for ADX smoothing
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema13_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        ema13 = ema13_1d_aligned[i]
        adx_val = adx_aligned[i]
        
        # Check EMA13 trend direction (using 3-bar momentum)
        if i >= 3:
            ema13_rising = ema13_1d_aligned[i] > ema13_1d_aligned[i-3]
            ema13_falling = ema13_1d_aligned[i] < ema13_1d_aligned[i-3]
        else:
            ema13_rising = False
            ema13_falling = False
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND ADX > 25 (strong trend) AND EMA13 rising
            if bull > 0 and adx_val > 25 and ema13_rising:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND ADX > 25 (strong trend) AND EMA13 falling
            elif bear < 0 and adx_val > 25 and ema13_falling:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on power reversal or weak trend
            # Exit if Bull Power <= 0 OR ADX < 20 (losing trend strength) OR EMA13 falling
            if bull <= 0 or adx_val < 20 or ema13_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on power reversal or weak trend
            # Exit if Bear Power >= 0 OR ADX < 20 OR EMA13 rising
            if bear >= 0 or adx_val < 20 or ema13_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals