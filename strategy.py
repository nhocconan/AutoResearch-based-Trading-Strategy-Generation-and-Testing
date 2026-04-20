#!/usr/bin/env python3
# 6h_RSIVolumeBreakout_With_1D_Trend_Filter
# Hypothesis: 6h RSI(14) breakout with volume confirmation, filtered by 1d EMA200 trend.
# In bull markets (price > 1d EMA200): long when RSI crosses above 50 with volume > 1.5x average.
# In bear markets (price < 1d EMA200): short when RSI crosses below 50 with volume > 1.5x average.
# RSI > 60 or < 40 prevents whipsaw in weak trends. Target: 50-150 total trades over 4 years.

name = "6h_RSIVolumeBreakout_With_1D_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # Wilder's smoothing for RSI
    period = 14
    if len(close) >= period + 1:
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full_like(close, np.nan)
    rsi = np.full_like(close, np.nan)
    valid = avg_loss != 0
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    rsi[valid] = 100 - (100 / (1 + rs[valid]))
    
    # Calculate average volume (20-period)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period + 1, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend from 1d EMA200
            uptrend = close[i] > ema200_1d_aligned[i]
            downtrend = close[i] < ema200_1d_aligned[i]
            
            # Volume filter: current volume > 1.5x average volume
            vol_filter = volume[i] > 1.5 * vol_ma[i]
            
            # Long: uptrend + RSI crosses above 50 + volume filter
            if uptrend and rsi[i] > 50 and rsi[i-1] <= 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + RSI crosses below 50 + volume filter
            elif downtrend and rsi[i] < 50 and rsi[i-1] >= 50 and vol_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend weakens or RSI overbought
            if (close[i] < ema200_1d_aligned[i] or 
                rsi[i] > 70 or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend weakens or RSI oversold
            if (close[i] > ema200_1d_aligned[i] or 
                rsi[i] < 30 or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals