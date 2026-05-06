#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12-hour Supertrend for trend direction and 1-hour
# Williams %R for entry timing. Enter long when Supertrend is bullish and Williams %R
# crosses above -80 (oversold bounce in uptrend). Enter short when Supertrend is bearish
# and Williams %R crosses below -20 (overbought rejection in downtrend).
# Uses 12h trend filter to avoid whipsaws, 1h momentum for precise entries.
# Target: 60-120 trades over 4 years (15-30/year) with 0.25 position sizing.

name = "6h_12hSupertrend_1hWilliamsR_Entry_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # First TR is just high-low
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    upperband = (high + low) / 2 + multiplier * atr
    lowerband = (high + low) / 2 - multiplier * atr
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
            
    return supertrend, direction

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R indicator."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    return wr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12-hour Supertrend ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Supertrend on 12h data
    st, direction = calculate_supertrend(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=10,
        multiplier=3.0
    )
    
    # Align Supertrend and direction to 6h timeframe
    st_aligned = align_htf_to_ltf(prices, df_12h, st)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate 1-hour Williams %R for entry timing
    df_1h = get_htf_data(prices, '1h')
    
    if len(df_1h) < 20:
        return np.zeros(n)
    
    wr = calculate_williams_r(
        df_1h['high'].values,
        df_1h['low'].values,
        df_1h['close'].values,
        period=14
    )
    
    # Align Williams %R to 6h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1h, wr)
    
    # Volume filter: above 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma_20
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(st_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(wr_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: 12h Supertrend bullish (direction=1) and Williams %R crosses above -80
            if (direction_aligned[i] == 1 and 
                wr_aligned[i] > -80 and 
                i > 50 and wr_aligned[i-1] <= -80 and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: 12h Supertrend bearish (direction=-1) and Williams %R crosses below -20
            elif (direction_aligned[i] == -1 and 
                  wr_aligned[i] < -20 and 
                  i > 50 and wr_aligned[i-1] >= -20 and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 12h Supertrend turns bearish
            if direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 12h Supertrend turns bullish
            if direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals