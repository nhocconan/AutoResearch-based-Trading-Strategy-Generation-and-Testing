#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation.
# Goes long when price breaks above R4 (resistance 4) with above-average volume,
# short when breaks below S4 (support 4) with volume.
# Uses 1d EMA50 as trend filter to avoid counter-trend trades.
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Camarilla levels are widely watched and provide strong support/resistance.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.

name = "exp_13811_6h_camarilla1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1 / 2  # Standard Camarilla uses 1.1
EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Pivot point
    pivot = (high + low + close) / 3
    # Range
    range_ = high - low
    # Resistance levels
    r1 = close + (range_ * CAMARILLA_MULTIPLIER * 1.0 / 12)
    r2 = close + (range_ * CAMARILLA_MULTIPLIER * 2.0 / 12)
    r3 = close + (range_ * CAMARILLA_MULTIPLIER * 3.0 / 12)
    r4 = close + (range_ * CAMARILLA_MULTIPLIER * 4.0 / 12)
    # Support levels
    s1 = close - (range_ * CAMARILLA_MULTIPLIER * 1.0 / 12)
    s2 = close - (range_ * CAMARILLA_MULTIPLIER * 2.0 / 12)
    s3 = close - (range_ * CAMARILLA_MULTIPLIER * 3.0 / 12)
    s4 = close - (range_ * CAMARILLA_MULTIPLIER * 4.0 / 12)
    return r1, r2, r3, r4, s1, s2, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and EMA trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate 1d EMA for trend filter
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    
    # Align 1d indicators to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend direction from 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Camarilla breakout signals
        long_signal = volume_ok and above_ema and close[i] > r4_aligned[i]
        short_signal = volume_ok and below_ema and close[i] < s4_aligned[i]
        
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
            # Exit long on close below S4 (strong support break)
            if close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above R4 (strong resistance break)
            if close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals