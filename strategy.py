#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Reversal_V1
Hypothesis: Price rejection at Camarilla R3/S3 levels with volume confirmation and ATR-based stoploss works on 12h timeframe for BTC and ETH in both bull and bear markets. Uses 1d timeframe for Camarilla calculation. Target: 15-30 trades/year per symbol (60-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First bar: use same values (will be filtered by min_periods later)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    camarilla_r3 = prev_close + range_1d * 1.1 / 4
    camarilla_s3 = prev_close - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: 20-period average on 12h timeframe
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average to reduce trades)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price rejects below S3 and closes back above it with volume
            if (prices['low'].iloc[i] < camarilla_s3_aligned[i] and 
                price > camarilla_s3_aligned[i]):
                if volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Short: price rejects above R3 and closes back below it with volume
            elif (prices['high'].iloc[i] > camarilla_r3_aligned[i] and 
                  price < camarilla_r3_aligned[i]):
                if volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price closes below S3 or ATR stoploss
            if price < camarilla_s3_aligned[i] or price < np.maximum.accumulate(close[:i+1])[-1] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R3 or ATR stoploss
            if price > camarilla_r3_aligned[i] or price > np.minimum.accumulate(close[:i+1])[-1] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_V1"
timeframe = "12h"
leverage = 1.0