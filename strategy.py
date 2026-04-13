#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian Channel (20) on 1d
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on 12h
    high_12h = high
    low_12h = low
    close_12h = close
    high_low_12h = high_12h - low_12h
    high_close_12h = np.abs(high_12h - np.roll(close_12h, 1))
    low_close_12h = np.abs(low_12h - np.roll(close_12h, 1))
    high_close_12h[0] = high_low_12h[0]
    low_close_12h[0] = high_low_12h[0]
    tr_12h = np.maximum(high_low_12h, np.maximum(high_close_12h, low_close_12h))
    atr_14_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * average volume
        vol_ma = np.mean(volume[max(0, i-20):i+1]) if i >= 20 else volume[i]
        vol_filter = volume[i] > 1.5 * vol_ma
        
        # Breakout conditions
        breakout_up = close[i] > donch_high_aligned[i]
        breakout_down = close[i] < donch_low_aligned[i]
        
        # ATR filter: volatility not too low
        vol_ok = atr_14_12h[i] > 0.01 * close[i]  # Avoid low volatility periods
        
        # Entry conditions
        long_entry = breakout_up and vol_filter and vol_ok
        short_entry = breakout_down and vol_filter and vol_ok
        
        # Exit conditions: opposite breakout or volatility collapse
        exit_long = position == 1 and (breakout_down or atr_14_12h[i] < 0.005 * close[i])
        exit_short = position == -1 and (breakout_up or atr_14_12h[i] < 0.005 * close[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_donchian_breakout_volume_filter"
timeframe = "12h"
leverage = 1.0