#!/usr/bin/env python3
# Hypothesis: 6h Supertrend (ATR=10, mult=3) with 1d weekly pivot filter and volume confirmation
# Long when: price > Supertrend (bullish), price > 1d weekly pivot R1, volume > 1.5x 20-period average
# Short when: price < Supertrend (bearish), price < 1d weekly pivot S1, volume > 1.5x 20-period average
# Exit when: Supertrend flips direction OR price crosses back below/above pivot R1/S1
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 25-50 trades/year.
# Designed to work in both bull (Supertrend + pivot breakout) and bear (Supertrend + pivot rejection) markets.

name = "6h_Supertrend_1dWeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Supertrend
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upperband = hl2 + (3 * atr)
    lowerband = hl2 - (3 * atr)
    
    supertrend = np.zeros_like(close)
    supertrend_direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    supertrend_direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
            supertrend_direction[i] = 1
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
            supertrend_direction[i] = -1
    
    # Get 1d data for weekly pivot points (using prior week's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's daily data
    # We'll use the prior week's high, low, close to calculate pivot for current week
    # For simplicity, we use rolling window of 5 days (1 week) to get weekly OHLC
    df_1d['week'] = pd.to_datetime(df_1d.index).isocalendar().week if hasattr(df_1d.index, 'weekday') else 0
    # Since we don't have actual dates in df_1d from get_htf_data, we approximate with rolling
    # In practice, get_htf_data returns actual dates, so we can use resample but we'll use rolling as approximation
    # Better: use actual date handling if available
    try:
        # Try to resample to weekly if index is datetime
        if isinstance(df_1d.index, pd.DatetimeIndex):
            weekly = df_1d.resample('W').agg({'high': 'max', 'low': 'min', 'close': 'last'})
            # Shift by 1 to use prior week's data
            weekly_shifted = weekly.shift(1)
            # Calculate pivot points
            weekly_shifted['pivot'] = (weekly_shifted['high'] + weekly_shifted['low'] + weekly_shifted['close']) / 3
            weekly_shifted['R1'] = 2 * weekly_shifted['pivot'] - weekly_shifted['low']
            weekly_shifted['S1'] = 2 * weekly_shifted['pivot'] - weekly_shifted['high']
            # Forward fill to get values for each day
            weekly_shifted = weekly_shifted.reindex(df_1d.index, method='ffill')
            pivot = weekly_shifted['pivot'].values
            R1 = weekly_shifted['R1'].values
            S1 = weekly_shifted['S1'].values
        else:
            # Fallback: use rolling window approximation
            high_5d = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max()
            low_5d = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min()
            close_5d = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last()
            pivot = (high_5d + low_5d + close_5d) / 3
            R1 = 2 * pivot - low_5d
            S1 = 2 * pivot - high_5d
            # Shift by 5 to use prior week's data
            pivot = np.concatenate([np.full(5, np.nan), pivot.values[:-5]])
            R1 = np.concatenate([np.full(5, np.nan), R1.values[:-5]])
            S1 = np.concatenate([np.full(5, np.nan), S1.values[:-5]])
    except:
        # Simple fallback: use single day's data (not ideal but functional)
        pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
        R1 = 2 * pivot - df_1d['low']
        S1 = 2 * pivot - df_1d['high']
        # Expand to match length and shift by 1 day
        pivot_vals = pivot.values
        R1_vals = R1.values
        S1_vals = S1.values
        pivot = np.concatenate([np.full(len(df_1d)-1, np.nan), pivot_vals[:-1]])
        R1 = np.concatenate([np.full(len(df_1d)-1, np.nan), R1_vals[:-1]])
        S1 = np.concatenate([np.full(len(df_1d)-1, np.nan), S1_vals[:-1]])
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Supertrend (bullish), price > R1, volume spike
            if (close[i] > supertrend[i] and 
                close[i] > R1_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Supertrend (bearish), price < S1, volume spike
            elif (close[i] < supertrend[i] and 
                  close[i] < S1_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Supertrend turns bearish OR price crosses below R1
            if (close[i] < supertrend[i]) or (close[i] < R1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Supertrend turns bullish OR price crosses above S1
            if (close[i] > supertrend[i]) or (close[i] > S1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals