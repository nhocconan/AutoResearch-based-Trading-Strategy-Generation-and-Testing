#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Camarilla pivot levels from 1d timeframe with volume confirmation and 6h momentum filter.
# Long when price breaks above Camarilla R4 level with above-average volume and RSI(14) > 50.
# Short when price breaks below Camarilla S4 level with above-average volume and RSI(14) < 50.
# Uses ATR-based stop loss to manage risk.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Camarilla levels provide clear support/resistance, RSI filters momentum direction, volume confirms breakout strength.

name = "exp_13871_6h_camarilla1d_rsi_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1 / 12  # Standard Camarilla multiplier
RSI_PERIOD = 14
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Daily range
    daily_range = high - low
    # Camarilla levels
    r4 = close + daily_range * CAMARILLA_MULTIPLIER * 4
    r3 = close + daily_range * CAMARILLA_MULTIPLIER * 3
    s3 = close - daily_range * CAMARILLA_MULTIPLIER * 3
    s4 = close - daily_range * CAMARILLA_MULTIPLIER * 4
    return r4, r3, s3, s4

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    
    # Load 1d data for Camarilla pivot levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align 1d Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data for RSI, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI for momentum filter
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]):
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
        
        # Momentum filter from RSI
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Camarilla breakout signals
        long_signal = volume_ok and rsi_bullish and close[i] > r4_1d_aligned[i]
        short_signal = volume_ok and rsi_bearish and close[i] < s4_1d_aligned[i]
        
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
            # Exit long on close below Camarilla S3 level
            if close[i] < s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above Camarilla R3 level
            if close[i] > r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals