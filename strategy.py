#!/usr/bin/env python3
# 4h_1d_volatility_breakout_v1
# Strategy: 4h volatility breakout with 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: In both bull and bear markets, volatility expansion precedes directional moves. 
# We identify periods of low volatility (contraction) using ATR ratio, then enter breakouts 
# in the direction of the 1d EMA trend. Volume confirms institutional participation. 
# This strategy avoids false breakouts in ranging markets by requiring volatility expansion.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for volatility measurement (using 4h data)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period ATR mean (volatility contraction/expansion)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    
    # Donchian channel (20-period) for breakout levels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility expansion: ATR ratio > 1.2 (volatility increasing)
        vol_expansion = atr_ratio[i] > 1.2
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_ratio[i] > 1.5
        
        # Entry conditions
        # Long: Volatility expansion + breakout above Donchian high + above 1d EMA50 (uptrend)
        if vol_expansion and vol_confirmed and close[i] > highest_high[i] and close[i] > ema_50_1d_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Volatility expansion + breakout below Donchian low + below 1d EMA50 (downtrend)
        elif vol_expansion and vol_confirmed and close[i] < lowest_low[i] and close[i] < ema_50_1d_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: volatility contraction or trend reversal
        elif position == 1 and (atr_ratio[i] < 0.8 or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (atr_ratio[i] < 0.8 or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals