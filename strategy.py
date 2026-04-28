#!/usr/bin/env python3
"""
6h_VWAP_MeanReversion_1dATRFilter
Hypothesis: Mean revert to VWAP when price deviates beyond 2*ATR(1d) with 1d trend filter. 
Works in both bull/bear by fading extremes in ranging markets while respecting higher timeframe trend.
Target: 25-35 trades/year per symbol with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(14) for deviation filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate VWAP on 6b data
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = vwap_num / vwap_den
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Deviation from VWAP in ATR units
        deviation = abs(close[i] - vwap[i]) / atr_1d_aligned[i]
        
        # Trend determination
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Mean reversion entry: price > 2*ATR from VWAP + counter-trend
        long_entry = (close[i] < vwap[i] and 
                     deviation > 2.0 and 
                     downtrend)  # Fade in downtrend
        
        short_entry = (close[i] > vwap[i] and 
                      deviation > 2.0 and 
                      uptrend)   # Fade in uptrend
        
        # Exit when price returns to VWAP or trend strengthens
        long_exit = close[i] >= vwap[i]
        short_exit = close[i] <= vwap[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_VWAP_MeanReversion_1dATRFilter"
timeframe = "6h"
leverage = 1.0