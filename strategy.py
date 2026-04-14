#!/usr/bin/env python3
"""
Hypothesis: 4-hour strategy using 12-hour Donchian breakout with volume confirmation and ATR filter.
Long when price breaks above 12h high + volume surge + ATR rising.
Short when price breaks below 12h low + volume surge + ATR rising.
Exit when price returns to 12h midpoint or ATR falls.
Designed for low turnover: ~20-40 trades/year per symbol to minimize fee drag.
Works in bull markets via breakouts and in bear via short-side symmetry.
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
    
    # Load 12-hour data once for Donchian channels and ATR
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12-hour Donchian channels (20)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 12-hour ATR (14) for volatility filter
    high_12h_arr = df_12h['high'].values
    low_12h_arr = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    tr1 = np.abs(high_12h_arr - low_12h_arr)
    tr2 = np.abs(high_12h_arr - np.concatenate([[close_12h_arr[0]], close_12h_arr[:-1]]))
    tr3 = np.abs(low_12h_arr - np.concatenate([[close_12h_arr[0]], close_12h_arr[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # 12-hour index
        idx_12h = i // 2  # 2 bars per day (4h timeframe)
        if idx_12h < 20:  # need enough for Donchian/ATR
            continue
        
        # Get previous 12h values to avoid look-ahead
        high_prev = donch_high[idx_12h - 1] if idx_12h - 1 < len(donch_high) else donch_high[-1]
        low_prev = donch_low[idx_12h - 1] if idx_12h - 1 < len(donch_low) else donch_low[-1]
        mid_prev = donch_mid[idx_12h - 1] if idx_12h - 1 < len(donch_mid) else donch_mid[-1]
        atr_prev = atr[idx_12h - 1] if idx_12h - 1 < len(atr) else atr[-1]
        if np.isnan(high_prev) or np.isnan(low_prev) or np.isnan(mid_prev) or np.isnan(atr_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        high_arr = np.full(len(df_12h), high_prev)
        low_arr = np.full(len(df_12h), low_prev)
        mid_arr = np.full(len(df_12h), mid_prev)
        atr_arr = np.full(len(df_12h), atr_prev)
        high_4h = align_htf_to_ltf(prices, df_12h, high_arr)[i]
        low_4h = align_htf_to_ltf(prices, df_12h, low_arr)[i]
        mid_4h = align_htf_to_ltf(prices, df_12h, mid_arr)[i]
        atr_4h = align_htf_to_ltf(prices, df_12h, atr_arr)[i]
        
        if position == 0:
            # Long: price breaks above 12h high + volume surge + ATR rising
            if (close[i] > high_4h and 
                volume[i] > vol_ma[i] * 1.5 and
                atr_4h > atr_4h * 0.95):  # ATR not falling sharply
                position = 1
                signals[i] = position_size
            # Short: price breaks below 12h low + volume surge + ATR rising
            elif (close[i] < low_4h and 
                  volume[i] > vol_ma[i] * 1.5 and
                  atr_4h > atr_4h * 0.95):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price returns to 12h mid or ATR falls significantly
            if close[i] < mid_4h or atr_4h < atr_4h * 0.8:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price returns to 12h mid or ATR falls significantly
            if close[i] > mid_4h or atr_4h < atr_4h * 0.8:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12h_Donchian_Volume_ATR"
timeframe = "4h"
leverage = 1.0