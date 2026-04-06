#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume confirmation + ADX trend filter.
# Uses 1d high/low/close to calculate Camarilla levels (resistance/support).
# Long when price crosses above L3 with volume confirmation in uptrend (ADX>25).
# Short when price crosses below H3 with volume confirmation in downtrend (ADX>25).
# Works in bull markets (buy strength at support) and bear markets (sell weakness at resistance).
# Low-frequency signals reduce fee drag while capturing meaningful moves.

name = "exp_13596_12h_camarilla1d_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1  # Standard Camarilla multiplier
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Camarilla levels based on previous day's range
    range_val = high - low
    # Resistance levels
    R4 = close + range_val * CAMARILLA_MULTIPLIER * 1.500
    R3 = close + range_val * CAMARILLA_MULTIPLIER * 1.250
    R2 = close + range_val * CAMARILLA_MULTIPLIER * 1.166
    R1 = close + range_val * CAMARILLA_MULTIPLIER * 1.083
    # Support levels
    S1 = close - range_val * CAMARILLA_MULTIPLIER * 1.083
    S2 = close - range_val * CAMARILLA_MULTIPLIER * 1.166
    S3 = close - range_val * CAMARILLA_MULTIPLIER * 1.250
    S4 = close - range_val * CAMARILLA_MULTIPLIER * 1.500
    return R1, R2, R3, R4, S1, S2, S3, S4

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Handle division by zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx.values

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
    
    # Load 1d data for Camarilla and ADX ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla for each day
    r1_vals = np.zeros(len(close_1d))
    r2_vals = np.zeros(len(close_1d))
    r3_vals = np.zeros(len(close_1d))
    r4_vals = np.zeros(len(close_1d))
    s1_vals = np.zeros(len(close_1d))
    s2_vals = np.zeros(len(close_1d))
    s3_vals = np.zeros(len(close_1d))
    s4_vals = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i > 0:  # Use previous day's data for today's levels
            r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        else:
            # For first day, use same day (will be filtered out by warmup anyway)
            r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r1_vals[i] = r1
        r2_vals[i] = r2
        r3_vals[i] = r3
        r4_vals[i] = r4
        s1_vals[i] = s1
        s2_vals[i] = s2
        s3_vals[i] = s3
        s4_vals[i] = s4
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_vals)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_vals)
    
    # Calculate 1d ADX for trend filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(volume_ma[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter from ADX
        strong_trend = adx_aligned[i] > ADX_THRESHOLD
        
        # Camarilla levels (using 1d data aligned to 12h)
        r3_val = r3_aligned[i]  # Resistance level 3
        h3_val = r3_aligned[i]  # Same as R3 for short
        l3_val = s3_aligned[i]  # Support level 3
        
        # Long signal: price crosses above L3 with volume in strong uptrend
        long_signal = volume_ok and strong_trend and close[i-1] <= l3_val and close[i] > l3_val
        
        # Short signal: price crosses below H3 with volume in strong downtrend
        short_signal = volume_ok and strong_trend and close[i-1] >= h3_val and close[i] < h3_val
        
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
            # Exit long on opposite signal or stop loss
            if close[i] < s1_aligned[i]:  # Exit if price breaks below S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite signal or stop loss
            if close[i] > r1_aligned[i]:  # Exit if price breaks above R1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals