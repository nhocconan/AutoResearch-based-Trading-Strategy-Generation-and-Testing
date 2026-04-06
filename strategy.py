#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13959_6d_camillia1d_pivot_vol"
timeframe = "6h"
leverage = 1.0

# Hypothesis: 6h Camarilla pivot reversal strategy with 1d volume confirmation.
# Uses 1d Camarilla levels (R3, S3, R4, S4) to identify extreme overbought/oversold conditions.
# Long when price breaks below S3 with volume confirmation (mean reversion bounce).
# Short when price breaks above R3 with volume confirmation (mean reversion fade).
# Uses R4/S4 as breakout levels for continuation in strong trends.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in ranging markets (reversions at R3/S3) and trending markets (breakouts at R4/S4).

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    typical = (high + low + close) / 3
    # Range
    range_val = high - low
    
    # Camarilla levels
    r4 = close + (range_val * 1.1 / 2)
    r3 = close + (range_val * 1.1 / 4)
    s3 = close - (range_val * 1.1 / 4)
    s4 = close - (range_val * 1.1 / 2)
    
    return r4, r3, s3, s4

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
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla and volume MA ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data for price, volume, and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or \
           np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Price levels
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        
        # Mean reversion signals at R3/S3
        long_signal = (close[i] < s3) and volume_ok  # Oversold bounce
        short_signal = (close[i] > r3) and volume_ok  # Overbought fade
        
        # Breakout signals at R4/S4 (continuation)
        breakout_long = (close[i] > r4) and volume_ok
        breakout_short = (close[i] < s4) and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            elif breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on mean reversion signal or stop
            if close[i] > r3 or close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on mean reversion signal or stop
            if close[i] < s3 or close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals