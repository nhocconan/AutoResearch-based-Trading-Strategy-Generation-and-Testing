#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot reversal with daily volume spike and weekly trend filter.
# Camarilla levels derived from previous day's range provide high-probability reversal zones.
# In ranging markets, price reverts to mean at these levels; in trending markets, breaks signal continuation.
# Volume surge confirms institutional interest. Weekly EMA ensures alignment with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag and improve generalization.

name = "exp_13282_12h_camarilla_pivot_vol_ema_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
EMA_PERIOD = 20  # Weekly EMA for trend filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # H = high, L = low, C = close of previous day
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, S1, S2, S3, S4
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    range_hl = H - L
    camarilla_r4 = C + (range_hl * CAMARILLA_MULTIPLIER / 2)
    camarilla_r3 = C + (range_hl * CAMARILLA_MULTIPLIER / 4)
    camarilla_r2 = C + (range_hl * CAMARILLA_MULTIPLIER / 6)
    camarilla_r1 = C + (range_hl * CAMARILLA_MULTIPLIER / 12)
    camarilla_s1 = C - (range_hl * CAMARILLA_MULTIPLIER / 12)
    camarilla_s2 = C - (range_hl * CAMARILLA_MULTIPLIER / 6)
    camarilla_s3 = C - (range_hl * CAMARILLA_MULTIPLIER / 4)
    camarilla_s4 = C - (range_hl * CAMARILLA_MULTIPLIER / 2)
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day to avoid look-ahead)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 12h indicators
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
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
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
        
        # Volume confirmation (need significant volume surge)
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Camarilla reversal signals with volume and trend filters
        # Long setup: price touches S3/S4 in uptrend with volume surge
        long_setup = volume_ok and uptrend and (
            low[i] <= s3_aligned[i] or low[i] <= s4_aligned[i]
        )
        # Short setup: price touches R3/R4 in downtrend with volume surge
        short_setup = volume_ok and downtrend and (
            high[i] >= r3_aligned[i] or high[i] >= r4_aligned[i]
        )
        
        # Generate signals
        if position == 0:
            if long_setup:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_setup:
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