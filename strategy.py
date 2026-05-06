#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day ATR-based volatility breakout with volume confirmation
# Long when price closes above previous close + ATR(10) with volume > 1.5x average
# Short when price closes below previous close - ATR(10) with volume > 1.5x average
# Uses volatility expansion to capture momentum bursts in both bull and bear markets
# Volume confirmation ensures breakouts have conviction
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "12h_1dATR10_Volume_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day ATR(10) for volatility breakout
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # True Range calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Dynamic breakout levels: previous close ± ATR(10)
    upper_break = np.roll(close_1d, 1) + atr_10
    lower_break = np.roll(close_1d, 1) - atr_10
    
    # Align breakout levels to 12h timeframe
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):  # Start after ATR warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_break_aligned[i]) or np.isnan(lower_break_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price closes above previous close + ATR(10) with volume confirmation
            if close[i] > upper_break_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below previous close - ATR(10) with volume confirmation
            elif close[i] < lower_break_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below previous close - ATR(10) (mean reversion)
            if close[i] < lower_break_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above previous close + ATR(10) (mean reversion)
            if close[i] > upper_break_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals