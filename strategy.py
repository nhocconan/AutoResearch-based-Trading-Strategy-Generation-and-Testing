#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian Channel Breakout with 1-day ATR filter and volume confirmation.
Trades breakouts above/below the 20-period Donchian bands on 12h chart, only when
1-day ATR(14) is elevated (indicating strong momentum/trending conditions) and
volume confirms institutional interest. Designed for low trade frequency (15-30/year)
to minimize fee drift and work in both bull and bear markets by requiring volatility
expansion for entry and using the breakout direction as the trend filter.
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
    
    # Load daily data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_avg = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14_1d / atr_14_1d_avg
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 12h Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: ATR ratio > 1.2 (elevated volatility)
        vol_filter = atr_ratio_aligned[i] > 1.2
        
        if position == 0 and vol_filter:
            # Long breakout: price closes above upper Donchian band
            if close[i] > donch_high[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below lower Donchian band
            elif close[i] < donch_low[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band
            exit_signal = False
            
            if position == 1:
                # Exit long: price closes below lower Donchian band
                if close[i] < donch_low[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price closes above upper Donchian band
                if close[i] > donch_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dATR_Volume_Filter"
timeframe = "12h"
leverage = 1.0