#!/usr/bin/env python3
"""
Experiment #11967: 6h Range Breakout with 1d Mean Reversion and Volume Confirmation
Hypothesis: In ranging markets (60-70% of time), price oscillates between support/resistance.
We identify 6d ranges using 6h high/low, then fade breaks that lack volume (false breakouts)
and trade breaks with volume (real breakouts). 1d RSI avoids trading against strong trends.
Works in bull/bear by adapting to range vs trend regimes. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11967_6h_range_breakout_1d_rsi_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
RANGE_LOOKBACK = 24  # 6d lookback for range (24 * 6h = 6d)
RSI_PERIOD = 14
VOLUME_LOOKBACK = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI for trend filter
    rsi_1d = calculate_rsi(df_1d['close'].values, RSI_PERIOD)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6d range (6 lookback periods)
    range_high = pd.Series(high).rolling(window=RANGE_LOOKBACK, min_periods=RANGE_LOOKBACK).max().values
    range_low = pd.Series(low).rolling(window=RANGE_LOOKBACK, min_periods=RANGE_LOOKBACK).min().values
    range_mid = (range_high + range_low) / 2
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RANGE_LOOKBACK, RSI_PERIOD, VOLUME_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if (np.isnan(range_high[i]) or np.isnan(range_low[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Range breakout detection
        breakout_above = close[i] > range_high[i-1] and close[i-1] <= range_high[i-1]
        breakout_below = close[i] < range_low[i-1] and close[i-1] >= range_low[i-1]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # 1d RSI filter: avoid extreme readings
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        # Entry logic: fade low-volume breaks, follow high-volume breaks
        if position == 0:
            # Fade low-volume breaks (mean reversion in range)
            if breakout_above and not volume_ok and rsi_not_overbought:
                signals[i] = -SIGNAL_SIZE  # short the breakout
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_below and not volume_ok and rsi_not_oversold:
                signals[i] = SIGNAL_SIZE  # long the breakout
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Follow high-volume breaks (trend continuation)
            elif breakout_above and volume_ok and rsi_not_overbought:
                signals[i] = SIGNAL_SIZE  # long the breakout
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_below and volume_ok and rsi_not_oversold:
                signals[i] = -SIGNAL_SIZE  # short the breakout
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