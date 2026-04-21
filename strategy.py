#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d trend filter. Elder Ray measures bull power (high-EMA) and bear power (low-EMA).
Go long when bull power > 0 and bear power rising (less negative), short when bear power < 0 and bull power falling (less positive).
Uses 13-period EMA for power calculation. 1d ADX > 20 filters for trending markets only.
Designed for 15-30 trades/year to minimize fee drag while capturing institutional momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX for trend filter
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    plus_dm[1:] = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm[1:] = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Elder Ray on 6h data: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA13 calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Smooth power signals with 3-period EMA to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Calculate derivatives (change) to detect rising/falling power
    bull_power_change = np.diff(bull_power_smooth, prepend=bull_power_smooth[0])
    bear_power_change = np.diff(bear_power_smooth, prepend=bear_power_smooth[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(adx_aligned[i]) or np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        bull_power_val = bull_power_smooth[i]
        bear_power_val = bear_power_smooth[i]
        bull_power_ch = bull_power_change[i]
        bear_power_ch = bear_power_change[i]
        
        if position == 0:
            # Enter long: bull power positive AND rising, bear power negative but rising (less negative)
            if (bull_power_val > 0 and 
                bull_power_ch > 0 and 
                bear_power_val < 0 and 
                bear_power_ch > 0 and 
                adx_val > 20):
                signals[i] = 0.25
                position = 1
            # Enter short: bear power negative AND falling, bull power positive but falling (less positive)
            elif (bear_power_val < 0 and 
                  bear_power_ch < 0 and 
                  bull_power_val > 0 and 
                  bull_power_ch < 0 and 
                  adx_val > 20):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: power signals reverse or ADX drops
            exit_signal = False
            
            if position == 1:
                # Exit long: bull power turns negative OR bear power turns positive
                if bull_power_val <= 0 or bear_power_val >= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: bear power turns positive OR bull power turns negative
                if bear_power_val >= 0 or bull_power_val <= 0:
                    exit_signal = True
            
            # Also exit if trend weakens
            if adx_val < 18:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dADX20_TrendFilter"
timeframe = "6h"
leverage = 1.0