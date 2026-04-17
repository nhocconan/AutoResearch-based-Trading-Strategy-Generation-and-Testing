#!/usr/bin/env python3
"""
12h_Williams_Alligator_RSI_Volume_v1
Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) + RSI(14) + Volume Spike on 12h timeframe.
Trend direction from Alligator alignment: Jaw > Teeth > Lips = uptrend, reverse for downtrend.
Entries on RSI extremes (overbought/oversold) with volume confirmation, exits on RSI normalization.
Designed to work in both bull and bear markets by combining trend-following with mean-reversion.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === 12h Williams Alligator (Smoothed Moving Average) ===
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (Prev SMMA * (period-1) + Close) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # === 12h RSI (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(len(close)):
        if i >= 14:
            if i == 14:
                avg_gain[i] = np.mean(gain[1:15])
                avg_loss[i] = np.mean(loss[1:15])
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        else:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine Alligator trend
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Jaw > Teeth > Lips
        is_uptrend = lips[i] > teeth[i] and teeth[i] > jaw[i]
        is_downtrend = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: uptrend AND RSI oversold (<30) AND volume confirmation
            if (is_uptrend and 
                rsi[i] < 30 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: downtrend AND RSI overbought (>70) AND volume confirmation
            elif (is_downtrend and 
                  rsi[i] > 70 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI overbought (>70) OR trend changes to downtrend
            if (rsi[i] > 70 or 
                not is_uptrend):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold (<30) OR trend changes to uptrend
            if (rsi[i] < 30 or 
                not is_downtrend):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_RSI_Volume_v1"
timeframe = "12h"
leverage = 1.0