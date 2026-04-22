#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Elder Ray Index (bull/bear power) + weekly pivot bias + volume confirmation.
# Uses weekly pivot points to establish bullish/bearish bias from higher timeframe.
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to detect strength.
# In bullish weekly bias: buy when bull power turns positive with volume.
# In bearish weekly bias: sell when bear power turns negative with volume.
# Designed to work in both bull and bear markets by aligning with weekly structure.
# Targets 15-35 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Determine weekly bias: above pivot = bullish, below pivot = bearish
    weekly_bias = np.where(close_1w > pivot_1w, 1, -1)  # 1=bullish, -1=bearish
    
    # Load daily data for EMA (used in Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align weekly bias and daily EMA to 6h timeframe
    bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # Calculate Elder Ray components on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Need to align EMA to 6h for Elder Ray calculation
    # We'll use the daily EMA aligned to 6h as proxy for 6h EMA (acceptable approximation)
    bull_power = high - ema_13_aligned  # Bull Power = High - EMA
    bear_power = low - ema_13_aligned   # Bear Power = Low - EMA
    
    # Calculate 6-period EMA of Elder Ray for smoothing (signal line)
    bull_power_smooth = pd.Series(bull_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Calculate volume average for spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bias_aligned[i]) or 
            np.isnan(ema_13_aligned[i]) or 
            np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bias = bias_aligned[i]
        bull = bull_power_smooth[i]
        bear = bear_power_smooth[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter based on weekly bias and Elder Ray signals
            if bias == 1:  # Weekly bullish bias
                # Look for bull power turning positive (bullish momentum)
                if bull > 0 and bull_power_smooth[i-1] <= 0 and vol_spike:
                    signals[i] = 0.25
                    position = 1
            elif bias == -1:  # Weekly bearish bias
                # Look for bear power turning negative (bearish momentum)
                if bear < 0 and bear_power_smooth[i-1] >= 0 and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bull power turns negative or weekly bias flips
                if bull < 0 or bias == -1:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bear power turns positive or weekly bias flips
                if bear > 0 or bias == 1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_WeeklyPivot_Bias"
timeframe = "6h"
leverage = 1.0