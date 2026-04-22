#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with 1-day ATR filter and volume confirmation.
Long when price breaks above upper band with 1-day ATR rising and volume spike.
Short when price breaks below lower band with 1-day ATR falling and volume spike.
Exit when price touches middle band (mean of upper/lower).
This strategy captures trend breakouts with volatility filtering to avoid false signals,
works in both bull and bear markets by following volatility regime.
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
    
    # Load 1-day data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_20
    lower_band = low_20
    middle_band = (upper_band + lower_band) / 2.0
    
    # 1-day ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = high_1d[0] - close_1d[0]  # First period
    tr3[0] = high_1d[0] - close_1d[0]  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1-day ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for ATR
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper band with 1-day ATR rising and volume spike
            if (close[i] > upper_band[i] and 
                atr_1d_aligned[i] > atr_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with 1-day ATR falling and volume spike
            elif (close[i] < lower_band[i] and 
                  atr_1d_aligned[i] < atr_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price touches middle band
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below middle band
                if close[i] < middle_band[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above middle band
                if close[i] > middle_band[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_DonchianBreakout_1dATR_Filter_Volume"
timeframe = "4h"
leverage = 1.0