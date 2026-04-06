#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + ADX filter.
# Uses Camarilla levels (support/resistance) for mean reversion in range-bound markets.
# Volume spike confirms institutional interest at key levels.
# ADX > 20 ensures we only trade when there's sufficient momentum.
# Designed to work in both bull and bear markets by fading extremes at pivot levels.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13565_12h_camarilla1d_volume_adx_v2"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ADX_PERIOD = 14
ADX_THRESHOLD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_adx(high, low, close, period):
    """Calculate ADX"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        elif low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.zeros_like(close)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, L4 (most important for intraday)
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    camarilla_h4 = close_1d + CAMARILLA_MULT * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - CAMARILLA_MULT * (high_1d - low_1d) / 2
    
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ADX
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]):
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
        
        # ADX filter: only trade when there's sufficient momentum
        adx_ok = adx[i] > ADX_THRESHOLD if not np.isnan(adx[i]) else False
        
        # Mean reversion signals at Camarilla levels
        # Long when price touches L4 support with volume and momentum
        long_signal = volume_ok and adx_ok and (low[i] <= camarilla_l4_aligned[i])
        # Short when price touches H4 resistance with volume and momentum
        short_signal = volume_ok and adx_ok and (high[i] >= camarilla_h4_aligned[i])
        
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