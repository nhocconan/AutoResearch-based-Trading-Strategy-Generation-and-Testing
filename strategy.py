#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_donchian_volume_breakout_v1
# Donchian(20) breakout on 12h timeframe with volume confirmation and 1d trend filter (EMA50).
# Works in both bull and bear markets by trading breakouts with trend alignment.
# Low trade frequency expected (15-30/year) due to strict breakout + volume + trend conditions.
name = "12h_1d_donchian_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        bullish_break = close[i] > highest_high[i-1] and vol_confirm
        bearish_break = close[i] < lowest_low[i-1] and vol_confirm
        
        # Trend filter: only take longs in uptrend (price > EMA50), shorts in downtrend (price < EMA50)
        bullish_signal = bullish_break and close[i] > ema_50_aligned[i]
        bearish_signal = bearish_break and close[i] < ema_50_aligned[i]
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = close[i] < ema_50_aligned[i]  # exit long when price crosses below EMA50
        exit_short = close[i] > ema_50_aligned[i]  # exit short when price crosses above EMA50
        
        if bullish_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals