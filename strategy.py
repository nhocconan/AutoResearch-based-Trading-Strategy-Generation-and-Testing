#!/usr/bin/env python3
"""
4h_Williams_Alligator_ElderRay_v1
Williams Alligator (Jaw/Teeth/Lips) + Elder Ray (Bull/Bear Power) for trend confirmation.
Long when price above Lips and Bull Power > 0; Short when price below Lips and Bear Power < 0.
Uses 12h timeframe for regime filter: only trade when 12h ADX > 25 (trending market).
Exit when price crosses back below/above Teeth or ADX < 20.
Designed to capture sustained trends with reduced whipsaw.
Target: 60-120 total trades over 4 years (15-30/year).
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
    
    # === Williams Alligator (13,8,5 SMAs shifted) ===
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    # Using SMA as approximation for SMMA (simple moving average)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # === Elder Ray (13-period EMA) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 12h ADX(14) for trend regime filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX on 12h data
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_12h * 14)
    minus_di_12h = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_12h * 14)
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
    adx_12h = pd.Series(dx_12h).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h ADX to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    
    # Warmup period - enough for all indicators
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above Lips, Bull Power > 0, 12h ADX > 25 (trending)
            if (close[i] > lips[i] and 
                bull_power[i] > 0 and 
                adx_12h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below Lips, Bear Power < 0, 12h ADX > 25 (trending)
            elif (close[i] < lips[i] and 
                  bear_power[i] < 0 and 
                  adx_12h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price below Teeth OR 12h ADX < 20 (losing trend)
            if (close[i] < teeth[i] or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above Teeth OR 12h ADX < 20 (losing trend)
            if (close[i] > teeth[i] or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_ElderRay_v1"
timeframe = "4h"
leverage = 1.0