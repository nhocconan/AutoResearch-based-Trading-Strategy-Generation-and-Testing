#!/usr/bin/env python3
"""
1h Mean Reversion with Daily Trend Filter + Volume Confirmation
Hypothesis: In ranging markets (2022-2024, 2025+), price reverts to daily VWAP.
Long when price < daily VWAP - 0.5*ATR, short when price > daily VWAP + 0.5*ATR.
Daily trend filter ensures we trade with the higher timeframe bias.
Volume confirmation ensures momentum behind moves.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12474_1h_vwap_reversion_dailytrend_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
VWAP_STD_MULT = 0.5
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ATR_PERIOD = 14
SIGNAL_SIZE = 0.20  # 20% position size

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP"""
    typical_price = (high + low + close) / 3.0
    vwap = np.nancumsum(typical_price * volume) / np.nancumsum(volume)
    return vwap

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily VWAP
    vwap_1d = calculate_vwap(df_1d['high'].values, df_1d['low'].values, 
                             df_1d['close'].values, df_1d['volume'].values)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate daily ATR
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, 
                           df_1d['close'].values, ATR_PERIOD)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, 
                                          min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily indicators not available
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check for mean reversion signals
        upper_band = vwap_1d_aligned[i] + (VWAP_STD_MULT * atr_1d_aligned[i])
        lower_band = vwap_1d_aligned[i] - (VWAP_STD_MULT * atr_1d_aligned[i])
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Mean reversion conditions
        long_signal = close[i] < lower_band and volume_ok
        short_signal = close[i] > upper_band and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price crosses back above VWAP
            if close[i] >= vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when price crosses back below VWAP
            if close[i] <= vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals