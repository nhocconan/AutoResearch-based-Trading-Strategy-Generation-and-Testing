#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12759_6d_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # use previous day
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
CAMARILLA_MULT = 1.5  # multiplier for entry/exit levels

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    # Camarilla levels
    r4 = close + range_ * 1.500
    r3 = close + range_ * 1.250
    r2 = close + range_ * 1.166
    r1 = close + range_ * 1.083
    s1 = close - range_ * 1.083
    s2 = close - range_ * 1.166
    s3 = close - range_ * 1.250
    s4 = close - range_ * 1.500
    return r1, r2, r3, r4, s1, s2, s3, s4, pivot

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
    
    r1 = np.full_like(close_1d, np.nan)
    r2 = np.full_like(close_1d, np.nan)
    r3 = np.full_like(close_1d, np.nan)
    r4 = np.full_like(close_1d, np.nan)
    s1 = np.full_like(close_1d, np.nan)
    s2 = np.full_like(close_1d, np.nan)
    s3 = np.full_like(close_1d, np.nan)
    s4 = np.full_like(close_1d, np.nan)
    pivot = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            # Cannot calculate for first day
            continue
        r1[i], r2[i], r3[i], r4[i], s1[i], s2[i], s3[i], s4[i], pivot[i] = calculate_camarilla_pivot(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
    
    # Align to 6h timeframe (use previous day's levels)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate 6h indicators
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
        if np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]):
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
        
        # Fade at R3/S3 with volume
        fade_short = volume_ok and close[i] >= r3_6h[i]
        fade_long = volume_ok and close[i] <= s3_6h[i]
        
        # Breakout continuation at R4/S4 with volume
        breakout_long = volume_ok and close[i] >= r4_6h[i]
        breakout_short = volume_ok and close[i] <= s4_6h[i]
        
        # Generate signals
        if position == 0:
            if fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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