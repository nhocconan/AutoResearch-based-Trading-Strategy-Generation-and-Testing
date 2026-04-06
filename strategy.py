#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean-reversion with 4h trend filter and 1d volume confirmation.
# In range-bound markets (2025-2026), price reverts to 4h VWAP with 1d volume spike.
# Uses 4h VWAP as dynamic mean, 1d volume > 1.5x 20-period MA for confirmation.
# Entry: price touches 4h VWAP ± 0.5*ATR(1h) with volume confirmation.
# Exit: reverse signal or stoploss at 2*ATR.
# Works in both bull/bear as mean reversion dominates in sideways markets.
# Target: 80-150 total trades over 4 years (20-38/year).

name = "exp_13354_1h_vwap_meanrev_4h_trend_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
VWAP_PERIOD = 20      # 4h VWAP lookback
VOLUME_MA_PERIOD = 20 # 1d volume MA
VOLUME_THRESHOLD = 1.5
ATR_PERIOD = 14
ATR_ENTRY_FACTOR = 0.5
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP using typical price"""
    tp = (high + low + close) / 3.0
    vwap_num = tp * volume
    vwap_den = volume
    vwap = pd.Series(vwap_num).rolling(window=period, min_periods=period).sum() / \
           pd.Series(vwap_den).rolling(window=period, min_periods=period).sum()
    return vwap.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load 4h data ONCE before loop for VWAP
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data ONCE before loop for volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h VWAP
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    vwap_4h = calculate_vwap(high_4h, low_4h, close_4h, volume_4h, VWAP_PERIOD)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Calculate 1d volume MA
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC (avoid Asian session noise)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if indicators not available
        if np.isnan(vwap_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or np.isnan(atr[i]):
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
        
        # Volume confirmation: 1d volume > threshold * MA
        volume_ok = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD)
        
        # Distance from 4h VWAP in ATR units
        vwap_dist = (close[i] - vwap_4h_aligned[i]) / atr[i]
        
        # Mean reversion signals
        long_signal = volume_ok and (vwap_dist <= -ATR_ENTRY_FACTOR)
        short_signal = volume_ok and (vwap_dist >= ATR_ENTRY_FACTOR)
        
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals