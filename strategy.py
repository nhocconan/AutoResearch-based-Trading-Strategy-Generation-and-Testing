#!/usr/bin/env python3
"""
6h Camarilla pivot strategy with volume confirmation and volatility filter.
Hypothesis: Price reverses at Camarilla support/resistance levels (S3/R3) with volume confirmation
and volatility filter to avoid low-momentum periods. Works in both bull and bear markets by
fading extremes and catching reversals. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14267_6h_camarilla_pivot_vol_volfilt_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels: R3, R4, S3, S4"""
    range_val = high - low
    close_val = close
    r3 = close_val + (range_val * 1.1 / 2)
    r4 = close_val + (range_val * 1.1)
    s3 = close_val - (range_val * 1.1 / 2)
    s4 = close_val - (range_val * 1.1)
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivot (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from daily data
    r3_1d, r4_1d, s3_1d, s4_1d = calculate_camarilla_pivot(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe (shifted by 1 day for completed bars only)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volatility filter: ATR(6) > 0.5 * ATR(24) to avoid low-momentum periods
    atr6 = calculate_atr(high, low, close, 6)
    atr24 = calculate_atr(high, low, close, 24)
    vol_filter = atr6 > (0.5 * atr24)
    
    # Volume filter: volume > 1.2x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter_volume = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 24 for volume, 6 for ATR)
    start = max(24, 6) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or \
           np.isnan(atr6[i]) or np.isnan(atr24[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility and volume filters must both be true
        if not (vol_filter[i] and vol_filter_volume[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Camarilla reversal signals
        # Long: price crosses above S3 with volume
        # Short: price crosses below R3 with volume
        long_signal = (close[i] > s3_1d_aligned[i]) and (close[i-1] <= s3_1d_aligned[i-1]) and vol_filter_volume[i]
        short_signal = (close[i] < r3_1d_aligned[i]) and (close[i-1] >= r3_1d_aligned[i-1]) and vol_filter_volume[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reversal at R3
            if close[i] >= r3_1d_aligned[i] or close[i] <= entry_price - (1.5 * atr6[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reversal at S3
            if close[i] <= s3_1d_aligned[i] or close[i] >= entry_price + (1.5 * atr6[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals