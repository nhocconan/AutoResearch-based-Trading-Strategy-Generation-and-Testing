#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with 12-hour Trend and Volume Confirmation.
Long when lips (fast SMA) > teeth (medium SMA) > jaws (slow SMA) and 12h EMA50 rising with volume spike.
Short when lips < teeth < jaws and 12h EMA50 falling with volume spike.
Exit when Alligator lines cross or 12h EMA50 reverses.
Williams Alligator identifies trend presence and direction; 12h EMA provides higher-timeframe trend filter;
volume spike confirms institutional participation. Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: 3 SMAs shifted forward
    # Jaws: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA = smoothed moving average (like EMA but with alpha = 1/period)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        sma = np.full_like(arr, np.nan, dtype=float)
        sma[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift jaws forward by 8, teeth by 5, lips by 3
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill leading NaNs from roll
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaws_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaws (bullish alignment) and 12h EMA50 rising with volume spike
            if (lips_shifted[i] > teeth_shifted[i] > jaws_shifted[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaws (bearish alignment) and 12h EMA50 falling with volume spike
            elif (lips_shifted[i] < teeth_shifted[i] < jaws_shifted[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross or 12h EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Lips <= Teeth or 12h EMA50 turns down
                if lips_shifted[i] <= teeth_shifted[i] or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Lips >= Teeth or 12h EMA50 turns up
                if lips_shifted[i] >= teeth_shifted[i] or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0