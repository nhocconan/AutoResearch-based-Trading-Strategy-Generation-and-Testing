#!/usr/bin/env python3
"""
Experiment #8191: 6-hour timeframe with 1-day HTF - Camarilla pivot levels with volume confirmation
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from daily timeframe
provide institutional-grade support/resistance levels. Price rejecting at R3/S3 with volume confirms mean reversion,
while breaking R4/S4 with volume indicates institutional breakout. This dual approach works in both trending 
and ranging markets by adapting to price action at key levels. Target 50-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8191_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    pivot = (high + low + close) / 3
    range_val = high - low
    
    # Camarilla levels
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    r2 = close + range_val * 1.1 / 6
    r1 = close + range_val * 1.1 / 12
    
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    
    return r1, r2, r3, r4, s1, s2, s3, s4, pivot

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data (1d) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for Camarilla levels
    r1 = np.full_like(close_1d, np.nan)
    r2 = np.full_like(close_1d, np.nan)
    r3 = np.full_like(close_1d, np.nan)
    r4 = np.full_like(close_1d, np.nan)
    s1 = np.full_like(close_1d, np.nan)
    s2 = np.full_like(close_1d, np.nan)
    s3 = np.full_like(close_1d, np.nan)
    s4 = np.full_like(close_1d, np.nan)
    pivot = np.full_like(close_1d, np.nan)
    
    # Calculate pivots for each day (starting from index 1 to use previous day)
    for i in range(1, len(close_1d)):
        r1[i], r2[i], r3[i], r4[i], s1[i], s2[i], s3[i], s4[i], pivot[i] = \
            calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    # Align Camarilla levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Price levels
        r3_level = r3_6h[i]
        s3_level = s3_6h[i]
        r4_level = r4_6h[i]
        s4_level = s4_6h[i]
        
        # Mean reversion signals at R3/S3
        # Long when price rejects S3 with volume
        long_mean_rev = (close[i] <= s3_level * 1.005 and  # Near S3
                         close[i] > s3_level and           # Above S3
                         volume_confirmed)
        
        # Short when price rejects R3 with volume
        short_mean_rev = (close[i] >= r3_level * 0.995 and  # Near R3
                          close[i] < r3_level and           # Below R3
                          volume_confirmed)
        
        # Breakout signals at R4/S4
        # Long when price breaks R4 with volume
        long_breakout = (close[i] >= r4_level and
                         volume_confirmed)
        
        # Short when price breaks S4 with volume
        short_breakout = (close[i] <= s4_level and
                          volume_confirmed)
        
        # Entry logic: mean reversion in range, breakout in trend
        # Simple approach: use both, let market decide
        if position == 0:
            if long_mean_rev or long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_mean_rev or short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals