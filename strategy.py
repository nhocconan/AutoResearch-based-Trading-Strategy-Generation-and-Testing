#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and moving averages
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on daily timeframe
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 21-period EMA on daily close
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align daily ATR and EMA to 6h timeframe
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_21_6h = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need daily EMA21 and ATR, 6h Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_6h[i]) or 
            np.isnan(ema_21_6h[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price above daily EMA21 AND breaks above 6h Donchian upper band
        long_condition = (close[i] > ema_21_6h[i]) and (close[i] > highest_high[i])
        
        # Short conditions: price below daily EMA21 AND breaks below 6h Donchian lower band
        short_condition = (close[i] < ema_21_6h[i]) and (close[i] < lowest_low[i])
        
        if position == 0:
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 6h Donchian lower band OR goes below daily EMA21
            if close[i] < lowest_low[i] or close[i] < ema_21_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 6h Donchian upper band OR goes above daily EMA21
            if close[i] > highest_high[i] or close[i] > ema_21_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_EMA21_Breakout"
timeframe = "6h"
leverage = 1.0