#!/usr/bin/env python3
"""
exp_6987_6h_camarilla1d_pivot_v2
Hypothesis: 6h Camarilla pivot levels from 1d data. Fade at R3/S3 (mean reversion in range), 
breakout continuation at R4/S4 (trend following). Uses 1w EMA200 as regime filter: 
only take longs in bull regime (price > 1w EMA200), shorts in bear regime (price < 1w EMA200).
Combines mean reversion and trend following based on market structure. Designed for 6h 
to capture swings with ~12-37 trades/year (50-150 total over 4 years).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6987_6h_camarilla1d_pivot_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's OHLC for Camarilla
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20  # ~5 days (6h bars)
EMA_WEEKLY_PERIOD = 200

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # Daily OHLC for pivots
    df_1w = get_htf_data(prices, '1w')  # Weekly for regime filter
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    r4 = pivot + (range_1d * 1.1)
    s4 = pivot - (range_1d * 1.1)
    
    # Align 1d levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate weekly EMA200 for regime filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_WEEKLY_PERIOD, adjust=False, min_periods=EMA_WEEKLY_PERIOD).mean().values
    ema_1w_6h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK + 1, EMA_WEEKLY_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1w_6h[i]) or np.isnan(r3_6h[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine regime from weekly EMA200
        bull_regime = close[i] > ema_1w_6h[i]
        bear_regime = close[i] < ema_1w_6h[i]
        
        # Camarilla-based signals
        # In bull regime: fade at S3/S4 (long), breakout at R4 (long)
        # In bear regime: fade at R3/R4 (short), breakout at S4 (short)
        long_signal = False
        short_signal = False
        
        if bull_regime:
            # Long opportunities in bull market
            if close[i] <= s3_6h[i]:  # Fade at S3 (mean reversion long)
                long_signal = True
            elif close[i] >= r4_6h[i]:  # Breakout at R4 (continuation long)
                long_signal = True
        elif bear_regime:
            # Short opportunities in bear market
            if close[i] >= r3_6h[i]:  # Fade at R3 (mean reversion short)
                short_signal = True
            elif close[i] <= s4_6h[i]:  # Breakout at S4 (continuation short)
                short_signal = True
        
        # Enter/exit positions
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long if opposite signal or stoploss/time
            if short_signal or close[i] >= r4_6h[i]:  # Take profit at R4 in bull
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short if opposite signal or stoploss/time
            if long_signal or close[i] <= s4_6h[i]:  # Take profit at S4 in bear
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals