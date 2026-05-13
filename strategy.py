#!/usr/bin/env python3
"""
4h_Chandelier_Exit_BullBear
Hypothesis: Use Chandelier Exit (trailing stop based on ATR) to ride trends while limiting drawdowns. Long when price closes above 22-period EMA and is above trailing long stop; short when price closes below 22-period EMA and is below trailing short stop. Uses 4h timeframe for trend capture and 1d for volatility context. Designed to work in both bull (riding uptrends) and bear (riding downtrends) markets with controlled risk via ATR-based stops.
"""

name = "4h_Chandelier_Exit_BullBear"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 22-period EMA for trend filter
    ema_22 = pd.Series(close).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    # Calculate ATR(22) for Chandelier Exit
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    # Get daily data for ATR multiplier context (optional volatility filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily ATR for volatility regime (not used in core logic but available for filtering)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).ewm(span=22, adjust=False, min_periods=22).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_avg)
    
    # Chandelier Exit multipliers
    mult = 3.0  # Standard multiplier for Chandelier Exit
    
    # Initialize trailing stops
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    
    # Calculate initial trailing stops
    for i in range(22, n):
        if np.isnan(atr[i]) or np.isnan(close[i-1]):
            long_stop[i] = np.nan
            short_stop[i] = np.nan
            continue
            
        if i == 22:
            # Initialize stops at first valid point
            long_stop[i] = high[i] - mult * atr[i]
            short_stop[i] = low[i] + mult * atr[i]
        else:
            # Update long stop: only move up, never down
            if close[i-1] > long_stop[i-1]:  # Was in long
                long_stop[i] = max(long_stop[i-1], high[i] - mult * atr[i])
            else:  # Was in short or flat
                long_stop[i] = high[i] - mult * atr[i]
                
            # Update short stop: only move down, never up
            if close[i-1] < short_stop[i-1]:  # Was in short
                short_stop[i] = min(short_stop[i-1], low[i] + mult * atr[i])
            else:  # Was in long or flat
                short_stop[i] = low[i] + mult * atr[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(22, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_22[i]) or np.isnan(long_stop[i]) or 
            np.isnan(short_stop[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price above EMA22 and above long stop (trailing buy stop)
            if close[i] > ema_22[i] and close[i] > long_stop[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price below EMA22 and below short stop (trailing sell stop)
            elif close[i] < ema_22[i] and close[i] < short_stop[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # MAINTAIN LONG: stay long if price above long stop
            if close[i] > long_stop[i]:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # MAINTAIN SHORT: stay short if price below short stop
            if close[i] < short_stop[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals