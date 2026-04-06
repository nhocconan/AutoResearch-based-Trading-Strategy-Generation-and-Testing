#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour strategy using daily pivot points (Camarilla-style) with volume confirmation and choppy market filter.
# Uses daily pivots for structure (support/resistance), volume spike for confirmation, and Choppiness Index for regime filtering.
# Designed to work in both bull and bear markets by fading extreme moves at pivot levels during ranging conditions.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Position size: 0.25 (25% of capital) to balance risk and return.

name = "exp_13736_12h_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's high/low/close
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 61.8  # Above this = ranging market (favor mean reversion)
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First period
    return tr

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (SMMA)"""
    tr = calculate_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    atr_sum = pd.Series(calculate_true_range(high, low, close)).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    # Handle division by zero or invalid cases
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    return chop

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for pivots and filters ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (Camarilla-style) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot levels based on previous day's range
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: support and resistance
    r4 = close_1d + (range_1d * 1.500)
    r3 = close_1d + (range_1d * 1.250)
    r2 = close_1d + (range_1d * 1.166)
    r1 = close_1d + (range_1d * 1.083)
    s1 = close_1d - (range_1d * 1.083)
    s2 = close_1d - (range_1d * 1.166)
    s3 = close_1d - (range_1d * 1.250)
    s4 = close_1d - (range_1d * 1.500)
    
    # Calculate ATR for 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate Choppiness Index for regime filter
    chop = calculate_choppiness(high, low, close, CHOPPINESS_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Align all daily data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK + 1, VOLUME_MA_PERIOD, CHOPPINESS_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(chop[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Choppy market condition (favor mean reversion in ranging markets)
        choppy_market = chop[i] > CHOPPINESS_THRESHOLD
        
        # Price proximity to pivot levels (within 0.5 * ATR)
        atr_buffer = 0.5 * atr[i]
        near_r1 = abs(close[i] - r1_aligned[i]) <= atr_buffer
        near_r2 = abs(close[i] - r2_aligned[i]) <= atr_buffer
        near_s1 = abs(close[i] - s1_aligned[i]) <= atr_buffer
        near_s2 = abs(close[i] - s2_aligned[i]) <= atr_buffer
        
        # Mean reversion signals in choppy markets
        if choppy_market and volume_ok:
            # Long near support levels
            long_signal = (near_s1 or near_s2) and close[i] > open[i]  # Bullish candle
            # Short near resistance levels
            short_signal = (near_r1 or near_r2) and close[i] < open[i]  # Bearish candle
        else:
            # In trending markets, wait for breakouts with volume
            long_signal = False
            short_signal = False
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long if price reaches resistance or stops
            if (near_r1 or near_r2) or close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short if price reaches support or stops
            if (near_s1 or near_s2) or close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals