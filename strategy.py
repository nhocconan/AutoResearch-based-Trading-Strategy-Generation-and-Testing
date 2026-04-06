#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12551_6d_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's high/low/close
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MIN_BARS_BETWEEN_TRADES = 4  # Minimum 6h bars between trades (~1 day)

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for the day
    Based on previous day's H/L/C
    R4 = C + ((H-L) * 1.1/2)
    R3 = C + ((H-L) * 1.1/4)
    R2 = C + ((H-L) * 1.1/6)
    R1 = C + ((H-L) * 1.1/12)
    S1 = C - ((H-L) * 1.1/12)
    S2 = C - ((H-L) * 1.1/6)
    S3 = C - ((H-L) * 1.1/4)
    S4 = C - ((H-L) * 1.1/2)
    """
    H = high
    L = low
    C = close
    
    R4 = C + ((H - L) * 1.1 / 2)
    R3 = C + ((H - L) * 1.1 / 4)
    S3 = C - ((H - L) * 1.1 / 4)
    S4 = C - ((H - L) * 1.1 / 2)
    
    return R3, S3, R4, S4

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R3_1d, S3_1d, R4_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe (shifted by 1 day for look-ahead prevention)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
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
    bars_since_last_trade = 0
    
    # Start from warmup period
    start = max(ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Enforce minimum bars between trades
        if bars_since_last_trade > 0:
            bars_since_last_trade += 1
        
        # Skip if daily data not available
        if np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]):
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
                bars_since_last_trade = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = -1
                bars_since_last_trade = 0
                continue
        
        # Fade at R3/S3, breakout continuation at R4/S4
        # Long: fade at S3, breakout above R4
        # Short: fade at R3, breakout below S4
        
        long_fade = close[i] <= S3_1d_aligned[i] and close[i] > S4_1d_aligned[i]
        long_breakout = close[i] > R4_1d_aligned[i]
        short_fade = close[i] >= R3_1d_aligned[i] and close[i] < R4_1d_aligned[i]
        short_breakout = close[i] < S4_1d_aligned[i]
        
        long_entry = long_fade or long_breakout
        short_entry = short_fade or short_breakout
        
        # Generate signals
        if position == 0 and bars_since_last_trade >= MIN_BARS_BETWEEN_TRADES:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_last_trade = 0
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_last_trade = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
        else:
            signals[i] = 0.0
    
    return signals