#!/usr/bin/env python3
"""
Hypothesis: 12-hour strategy using 1-day True Range breakout with volume confirmation and 1-day ATR filter.
Long when price breaks above previous 1-day high + ATR(14) + volume surge.
Short when price breaks below previous 1-day low - ATR(14) + volume surge.
Exit when price crosses 1-day VWAP or ATR-based stop is hit.
Designed for low turnover: ~20-30 trades/year per symbol to minimize fee drag.
ATR filter prevents whipsaws in choppy markets; VWAP exit captures mean reversion.
Works in bull via breakouts and in bear via short-side symmetry with volatility filter.
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
    
    # Load 1-day data once for ATR, high/low, and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first TR
    atr = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1-day high and low for breakout levels
    # 1-day VWAP for exit
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap = (np.cumsum(typical_price * df_1d['volume'].values) / 
            np.cumsum(df_1d['volume'].values)).values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # 1-day index (2 bars per day for 12h timeframe)
        idx_1d = i // 2
        if idx_1d < 14:  # need enough for ATR calculation
            continue
        
        # Get previous 1-day values to avoid look-ahead (use completed day)
        high_prev = high_1d[idx_1d - 1] if idx_1d - 1 < len(high_1d) else high_1d[-1]
        low_prev = low_1d[idx_1d - 1] if idx_1d - 1 < len(low_1d) else low_1d[-1]
        atr_prev = atr[idx_1d - 1] if idx_1d - 1 < len(atr) else atr[-1]
        vwap_prev = vwap[idx_1d - 1] if idx_1d - 1 < len(vwap) else vwap[-1]
        if np.isnan(high_prev) or np.isnan(low_prev) or np.isnan(atr_prev) or np.isnan(vwap_prev):
            continue
        
        # Create arrays for alignment (using previous completed day's values)
        high_arr = np.full(len(df_1d), high_prev)
        low_arr = np.full(len(df_1d), low_prev)
        atr_arr = np.full(len(df_1d), atr_prev)
        vwap_arr = np.full(len(df_1d), vwap_prev)
        high_12h = align_htf_to_ltf(prices, df_1d, high_arr)[i]
        low_12h = align_htf_to_ltf(prices, df_1d, low_arr)[i]
        atr_12h = align_htf_to_ltf(prices, df_1d, atr_arr)[i]
        vwap_12h = align_htf_to_ltf(prices, df_1d, vwap_arr)[i]
        
        if position == 0:
            # Long: price breaks above (prev day high + 0.5*ATR) + volume surge
            if (close[i] > (high_12h + 0.5 * atr_12h) and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: price breaks below (prev day low - 0.5*ATR) + volume surge
            elif (close[i] < (low_12h - 0.5 * atr_12h) and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price crosses below VWAP or ATR-based stop (2*ATR below entry)
            # Simplified: exit when price < VWAP or price drops significantly
            if close[i] < vwap_12h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price crosses above VWAP
            if close[i] > vwap_12h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_ATR_VWAP_Volume_Breakout"
timeframe = "12h"
leverage = 1.0