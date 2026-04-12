#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d ATR for volatility regime detection (using previous day's ATR)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    
    # 1d ATR(10) - previous day's value to avoid look-ahead
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    atr_1d_prev = np.roll(atr_1d, 1)  # Use previous day's ATR
    atr_1d_prev[0] = np.nan  # First value invalid
    
    # 1d ATR(40) for volatility ratio
    atr_40_1d = pd.Series(tr_1d).rolling(window=40, min_periods=40).mean().values
    atr_40_1d_prev = np.roll(atr_40_1d, 1)
    atr_40_1d_prev[0] = np.nan
    
    # Volatility ratio: low when ATR(10)/ATR(40) < 0.6 (low volatility regime)
    vol_ratio = atr_1d_prev / atr_40_1d_prev
    low_vol_1d = vol_ratio < 0.6
    
    # Align 1d volatility regime to 4h
    low_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, low_vol_1d)
    
    # 4h Donchian channel breakout (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Entry signals: breakout in low volatility regime
    long_breakout = close > highest_20
    short_breakout = close < lowest_20
    
    # Exit signals: return to midpoint of Donchian channel
    midpoint_20 = (highest_20 + lowest_20) / 2.0
    long_exit = close < midpoint_20
    short_exit = close > midpoint_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after sufficient lookback
        # Skip if volatility data not ready
        if np.isnan(low_vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Check for entry signals (only in low volatility regime)
        long_signal = long_breakout[i] and low_vol_1d_aligned[i]
        short_signal = short_breakout[i] and low_vol_1d_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif long_exit and position == 1:
            position = 0
            signals[i] = 0.0
        elif short_exit and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals