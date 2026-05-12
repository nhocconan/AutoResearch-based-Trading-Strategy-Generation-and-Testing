#!/usr/bin/env python3
# 4h_1d_Pivot_Reversion
# Hypothesis: Mean reversion from daily pivot points (PP, R1, S1) with 4h execution.
# Uses 1d pivot levels calculated from prior day's OHLC. Enters when 4h price touches
# pivot support/resistance with rejection candle (pin bar) and volume confirmation.
# Works in ranging markets (2025-2026) and pulls back in trends (2021-2024).
# Low frequency: targets 20-40 trades/year by requiring confluence of pivot, price action, and volume.

name = "4h_1d_Pivot_Reversion"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.3 * vol_ma)
    
    # Daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 4h timeframe (use previous day's pivot for current day)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Pin bar detection: small body, long wick
    body = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    
    # Bullish pin: long lower wick, small body
    bullish_pin = (lower_wick > 2 * body) & (body < (high - low) * 0.3)
    # Bearish pin: long upper wick, small body
    bearish_pin = (upper_wick > 2 * body) & (body < (high - low) * 0.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches S1 with bullish pin + volume
            if (low[i] <= s1_aligned[i] * 1.001 and  # Allow small slippage
                bullish_pin[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R1 with bearish pin + volume
            elif (high[i] >= r1_aligned[i] * 0.999 and   # Allow small slippage
                  bearish_pin[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot or shows bearish rejection
            if (close[i] >= pivot_aligned[i] * 0.999 or 
                bearish_pin[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or shows bullish rejection
            if (close[i] <= pivot_aligned[i] * 1.001 or 
                bullish_pin[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

open_ = prices['open'].values  # Moved after price array extraction to avoid forward reference