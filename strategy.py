#!/usr/bin/env python3
# 4h_VolatilityBreakout_With_VolumeAndTrendFilter
# Hypothesis: Breakouts from volatility contraction (low ATR) combined with volume surge and 1d EMA trend filter.
# In uptrend (price > 1d EMA50): long when ATR ratio < 0.6 and volume > 1.5x 20-period average.
# In downtrend (price < 1d EMA50): short when ATR ratio < 0.6 and volume > 1.5x 20-period average.
# ATR ratio = ATR(7) / ATR(30) - measures volatility contraction.
# Exit when ATR ratio > 1.2 (volatility expansion) or trend reverses.
# Designed for low trade frequency (<40/year) to minimize fee drag in 4h timeframe.

name = "4h_VolatilityBreakout_With_VolumeAndTrendFilter"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(7) and ATR(30) for volatility ratio
    def calculate_atr(high, low, close, period):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = np.full_like(high, np.nan)
        if len(high) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period + 1, len(high)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr
    
    atr7 = calculate_atr(high, low, close, 7)
    atr30 = calculate_atr(high, low, close, 30)
    atr_ratio = np.where(atr30 != 0, atr7 / atr30, np.nan)
    
    # Calculate volume ratio (current volume / 20-period average volume)
    vol_ma20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma20[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma20[i] = (vol_ma20[i-1] * 19 + volume[i]) / 20
    volume_ratio = np.where(vol_ma20 != 0, volume / vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure we have enough data for ATR(30) and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend from 1d EMA50
            uptrend = close[i] > ema50_1d_aligned[i]
            downtrend = close[i] < ema50_1d_aligned[i]
            
            # Volatility breakout conditions: low volatility + volume surge
            low_volatility = atr_ratio[i] < 0.6
            volume_surge = volume_ratio[i] > 1.5
            
            # Long: uptrend + low volatility + volume surge
            if uptrend and low_volatility and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + low volatility + volume surge
            elif downtrend and low_volatility and volume_surge:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if volatility expands or trend reverses
            if atr_ratio[i] > 1.2 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if volatility expands or trend reverses
            if atr_ratio[i] > 1.2 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals