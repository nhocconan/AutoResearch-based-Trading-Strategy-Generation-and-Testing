#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Breakouts above recent 4h high/low with volume confirmation and alignment with 1d trend capture strong moves in both bull and bear markets.
# Long when price > 4h high (lookback 20) + volume > 1.5x 10-period average + 1d close > 1d EMA50.
# Short when price < 4h low (lookback 20) + volume > 1.5x 10-period average + 1d close < 1d EMA50.
# Exit on opposite signal or trend reversal. Position size: ±0.25.
# Uses 4h for entry/exit and 1d for trend filter and breakout levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (10-period MA on 4h)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Get 1d data for trend filter and breakout levels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h rolling high (20 periods) and low (20 periods)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    high_20 = high_series.rolling(window=20, min_periods=20).max().values
    low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(10, 20, 50)  # volume MA10, 4h high/low lookback, EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma10[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-period average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        if position == 0:
            # Long: price > 4h high (20) + volume filter + 1d uptrend
            if close[i] > high_20[i] and volume_filter and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < 4h low (20) + volume filter + 1d downtrend
            elif close[i] < low_20[i] and volume_filter and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < 4h low (20) or 1d trend turns down
            if close[i] < low_20[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > 4h high (20) or 1d trend turns up
            if close[i] > high_20[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_HighLowBreakout_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0