#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
Williams Alligator (JAW/TEETH/LIPS) identifies trendless markets when lines are intertwined.
Entry occurs when LIPS crosses above/below TEETH/JAW with EMA50 trend alignment and volume spike.
Designed for 4h timeframe to capture medium-term trends while avoiding choppy markets.
Target: 19-50 trades/year per symbol (75-200 total over 4 years).
Uses discrete position sizing (0.30) to balance return and fee drag.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 4h timeframe
    # JAW (Blue): 13-period SMMA, shifted 8 bars ahead
    # TEETH (Red): 8-period SMMA, shifted 5 bars ahead  
    # LIPS (Green): 5-period SMMA, shifted 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Calculate volume spike: current volume > 1.8x 30-period MA
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > 1.8 * vol_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 13+8, 8+5, 5+3)  # EMA50, vol MA30, and Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA50 = uptrend, close < 1d EMA50 = downtrend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Alligator conditions
        # Lips above Teeth and Jaw = bullish alignment
        bullish_alignment = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
        # Lips below Teeth and Jaw = bearish alignment  
        bearish_alignment = lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment AND uptrend AND volume spike
            if bullish_alignment and trend_up and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: Bearish Alligator alignment AND downtrend AND volume spike
            elif bearish_alignment and trend_down and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Alligator lines become intertwined (market going choppy) OR trend reversal
            exit_signal = False
            if position == 1:
                # Exit long on bearish alignment or trend reversal
                if bearish_alignment or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Exit short on bullish alignment or trend reversal
                if bullish_alignment or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0