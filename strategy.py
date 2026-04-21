#!/usr/bin/env python3
"""
1h_Camarilla_Pivot_Breakout_VolumeFilter_V1
Hypothesis: Camarilla pivot R1/S1 breakout with volume confirmation (>1.5x 20-bar MA) on 1h timeframe. Uses 4h timeframe for trend filter (price > EMA34 for longs, price < EMA34 for shorts). Target: 15-35 trades/year per symbol (60-140 over 4 hours). Works in both bull and bear markets by only taking breakouts in the direction of the 4h trend, avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data once for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 4h timeframe for trend
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate Camarilla pivots on 1h timeframe (using previous bar's OHLC)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Camarilla levels: based on previous bar's range
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # We use previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # 4h trend filter
        uptrend = price > ema34_4h_aligned[i]
        downtrend = price < ema34_4h_aligned[i]
        
        if position == 0:
            # Long: break above R1 in uptrend with volume
            if price > R1[i] and uptrend and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 in downtrend with volume
            elif price < S1[i] and downtrend and volume_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price crosses below previous bar's close or stoploss
            if price < prev_close[i] or price < prices['close'].iloc[i-1] - 2.0 * (
                    np.abs(high[i] - low[i]) + np.abs(high[i] - np.roll(close, 1)[i]) + 
                    np.abs(low[i] - np.roll(close, 1)[i]))/3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price crosses above previous bar's close or stoploss
            if price > prev_close[i] or price > prices['close'].iloc[i-1] + 2.0 * (
                    np.abs(high[i] - low[i]) + np.abs(high[i] - np.roll(close, 1)[i]) + 
                    np.abs(low[i] - np.roll(close, 1)[i]))/3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_Pivot_Breakout_VolumeFilter_V1"
timeframe = "1h"
leverage = 1.0