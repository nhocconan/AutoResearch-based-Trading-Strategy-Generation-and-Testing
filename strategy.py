#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot from 1d with volume confirmation.
# Uses 1d Camarilla levels (R3, R4, S3, S4) for mean reversion and breakouts.
# In ranging markets: fade at S3/R3 with TP at S2/R2. In trending markets: breakout at S4/R4 with continuation.
# Volume filter ensures institutional participation. Works in both bull (breakouts) and bear (mean reversion).
# Target: 80-160 total trades over 4 years (20-40/year) with size 0.25.

name = "exp_13491_6h_camarilla_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
PROFIT_TAKE_LEVEL = 0.5  # Take profit at 50% of range

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close
    c = close + (range_val * 1.1 / 12)
    r3 = close + (range_val * 1.1 / 6)
    r4 = close + (range_val * 1.1 / 4)
    s3 = close - (range_val * 1.1 / 6)
    s4 = close - (range_val * 1.1 / 4)
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3_1d, r4_1d, s3_1d, s4_1d = [], [], [], []
    for i in range(len(high_1d)):
        r3, r4, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r3_1d.append(r3)
        r4_1d.append(r4)
        s3_1d.append(s3)
        s4_1d.append(s4)
    
    r3_1d = np.array(r3_1d)
    r4_1d = np.array(r4_1d)
    s3_1d = np.array(s3_1d)
    s4_1d = np.array(s4_1d)
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    profit_taken = False
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]):
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
                profit_taken = False
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                profit_taken = False
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla levels
        r3 = r3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Mean reversion signals (fade at S3/R3)
        mean_revert_long = volume_ok and (close[i] <= s3) and (low[i] < s3)
        mean_revert_short = volume_ok and (close[i] >= r3) and (high[i] > r3)
        
        # Breakout signals (continue at S4/R4)
        breakout_long = volume_ok and (close[i] > r4) and (open := high if i==0 else high[i-1]) <= r4 and high[i] > r4
        breakout_short = volume_ok and (close[i] < s4) and (open := low if i==0 else low[i-1]) >= s4 and low[i] < s4
        
        # Generate signals
        if position == 0:
            if mean_revert_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                profit_taken = False
            elif mean_revert_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                profit_taken = False
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                profit_taken = False
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                profit_taken = False
            else:
                signals[i] = 0.0
        elif position == 1:
            # Take profit at 50% of range from S3 to R3
            range_val = r3 - s3
            profit_level = s3 + (range_val * PROFIT_TAKE_LEVEL)
            if close[i] >= profit_level and not profit_taken:
                signals[i] = SIGNAL_SIZE * 0.5  # Half position
                profit_taken = True
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Take profit at 50% of range from S3 to R3
            range_val = r3 - s3
            profit_level = r3 - (range_val * PROFIT_TAKE_LEVEL)
            if close[i] <= profit_level and not profit_taken:
                signals[i] = -SIGNAL_SIZE * 0.5  # Half position
                profit_taken = True
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals