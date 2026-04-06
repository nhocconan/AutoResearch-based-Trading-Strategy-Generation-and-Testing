#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12772_12h_camarilla1d_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper handling"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Set first element to 0 (no previous close)
    tr[0] = 0
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla levels for given period"""
    # Camarilla formula based on previous day's OHLC
    range_val = high - low
    close_val = close
    
    # Resistance levels
    r4 = close_val + range_val * 1.1 / 2
    r3 = close_val + range_val * 1.1 / 4
    r2 = close_val + range_val * 1.1 / 6
    r1 = close_val + range_val * 1.1 / 12
    
    # Support levels
    s1 = close_val - range_val * 1.1 / 12
    s2 = close_val - range_val * 1.1 / 6
    s3 = close_val - range_val * 1.1 / 4
    s4 = close_val - range_val * 1.1 / 2
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use only completed daily bars (no look-ahead)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d, 1)
    # Set first element to NaN (no previous day)
    high_1d_shifted[0] = np.nan
    low_1d_shifted[0] = np.nan
    close_1d_shifted[0] = np.nan
    
    # Calculate Camarilla levels for each day
    r1_1d = np.full_like(high_1d, np.nan)
    r2_1d = np.full_like(high_1d, np.nan)
    r3_1d = np.full_like(high_1d, np.nan)
    r4_1d = np.full_like(high_1d, np.nan)
    s1_1d = np.full_like(high_1d, np.nan)
    s2_1d = np.full_like(high_1d, np.nan)
    s3_1d = np.full_like(high_1d, np.nan)
    s4_1d = np.full_like(high_1d, np.nan)
    
    for i in range(len(high_1d)):
        if not np.isnan(high_1d_shifted[i]) and not np.isnan(low_1d_shifted[i]) and not np.isnan(close_1d_shifted[i]):
            r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
                high_1d_shifted[i], low_1d_shifted[i], close_1d_shifted[i]
            )
            r1_1d[i] = r1
            r2_1d[i] = r2
            r3_1d[i] = r3
            r4_1d[i] = r4
            s1_1d[i] = s1
            s2_1d[i] = s2
            s3_1d[i] = s3
            s4_1d[i] = s4
    
    # Align to 12h timeframe (2 bars per day)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla level touches with volume
        # Long when price touches S3 or S4 with volume
        touch_long = volume_ok and (close[i] <= s3_12h[i] * 1.001 or close[i] <= s4_12h[i] * 1.001)
        # Short when price touches R3 or R4 with volume
        touch_short = volume_ok and (close[i] >= r3_12h[i] * 0.999 or close[i] >= r4_12h[i] * 0.999)
        
        # Generate signals
        if position == 0:
            if touch_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif touch_short:
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