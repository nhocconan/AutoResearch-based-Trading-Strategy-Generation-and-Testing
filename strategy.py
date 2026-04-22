#!/usr/bin/env python3

"""
Hypothesis: 12-hour Camarilla pivot reversal with 1-day trend filter and volume confirmation.
Trades reversals at Camarilla S3/R3 levels in the direction of the 1d EMA trend.
Uses volume spike confirmation to avoid false signals. Designed for low trade frequency
(12-37 trades/year) to minimize fee flood and work in both bull and bear markets by
aligning with higher timeframe trend. Camarilla levels provide high-probability
reversal zones, especially effective in ranging markets common in 2025.
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
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 12h data for Camarilla calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_S3 = np.full(len(df_12h), np.nan)
    camarilla_R3 = np.full(len(df_12h), np.nan)
    
    for i in range(len(df_12h)):
        if i >= 1:  # Need previous bar's range
            high_prev = high_12h[i-1]
            low_prev = low_12h[i-1]
            close_prev = close_12h[i-1]
            range_prev = high_prev - low_prev
            
            camarilla_S3[i] = close_prev - 1.1 * range_prev / 6
            camarilla_R3[i] = close_prev + 1.1 * range_prev / 6
    
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R3)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price at S3 support in uptrend
            if close[i] <= camarilla_S3_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price at R3 resistance in downtrend
            elif close[i] >= camarilla_R3_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price moves to opposite Camarilla level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches R3 or closes below 1d EMA
                if close[i] >= camarilla_R3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches S3 or closes above 1d EMA
                if close[i] <= camarilla_S3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_S3R3_Reversal_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0