#!/usr/bin/env python3
"""
Experiment #12075: 6h Camarilla Pivot Reversal with 1w Trend and Volume Confirmation
Hypothesis: Camarilla pivot levels from daily data identify key support/resistance. 
Fade at R3/S3 (mean reversion) and breakout continuation at R4/S4 (trend following) 
with 1-week EMA trend filter and volume confirmation. Works in ranging and trending markets.
Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12075_6h_camarilla_pivot_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1  # Standard Camarilla multiplier
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r4 = pivot + (range_ * CAMARILLA_MULTIPLIER * 1.1 / 2)
    r3 = pivot + (range_ * CAMARILLA_MULTIPLIER * 1.1 / 4)
    s3 = pivot - (range_ * CAMARILLA_MULTIPLIER * 1.1 / 4)
    s4 = pivot - (range_ * CAMARILLA_MULTIPLIER * 1.1 / 2)
    return r4, r3, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 1d data for Camarilla pivots and 1w data for trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4, camarilla_r3, camarilla_s3, camarilla_s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate 1w EMA for trend
    ema_1w = calculate_ema(df_1w['close'].values, TREND_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 1  # Need previous bar for Camarilla levels
    
    for i in range(start, n):
        # Skip if 1w EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        
        # Get previous day's Camarilla levels (use previous day's data)
        prev_idx = i - 1
        if prev_idx < 0 or np.isnan(camarilla_r4[prev_idx]) or np.isnan(camarilla_r3[prev_idx]) or \
           np.isnan(camarilla_s3[prev_idx]) or np.isnan(camarilla_s4[prev_idx]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        r4 = camarilla_r4[prev_idx]
        r3 = camarilla_r3[prev_idx]
        s3 = camarilla_s3[prev_idx]
        s4 = camarilla_s4[prev_idx]
        
        # Volume confirmation
        volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (1w)
        uptrend_1w = close[i] > ema_1w_aligned[i]
        downtrend_1w = close[i] < ema_1w_aligned[i]
        
        # Entry conditions
        # Fade at S3/R3 (mean reversion): long at S3 bounce, short at R3 rejection
        long_fade = (low[i] <= s3 and close[i] > s3) and volume_ok and downtrend_1w
        short_fade = (high[i] >= r3 and close[i] < r3) and volume_ok and uptrend_1w
        
        # Breakout continuation at S4/R4 (trend following): break S4 for short, break R4 for long
        long_breakout = (high[i] > r4 and close[i] > r4) and volume_ok and uptrend_1w
        short_breakout = (low[i] < s4 and close[i] < s4) and volume_ok and downtrend_1w
        
        long_entry = long_fade or long_breakout
        short_entry = short_fade or short_breakout
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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