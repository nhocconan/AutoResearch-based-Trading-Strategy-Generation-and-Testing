#!/usr/bin/env python3
"""
4h Bollinger Band Squeeze + Volume Confirmation + ATR Stop
Hypothesis: Bollinger Band squeeze identifies low volatility periods; breakout with volume
captures directional moves. Works in bull (breakouts with volume) and bear (breakdowns with volume).
ATR stop-loss limits drawdown. Target: 100-200 trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bb_squeeze_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    basis = np.full(n, np.nan)
    dev = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n >= bb_length:
        for i in range(bb_length-1, n):
            basis[i] = np.mean(close[i-bb_length+1:i+1])
            dev[i] = bb_mult * np.std(close[i-bb_length+1:i+1])
            upper[i] = basis[i] + dev[i]
            lower[i] = basis[i] - dev[i]
    
    # Bollinger Band Width (normalized)
    bb_width = np.full(n, np.nan)
    if n >= bb_length:
        for i in range(bb_length-1, n):
            if not np.isnan(basis[i]) and basis[i] != 0:
                bb_width[i] = (upper[i] - lower[i]) / basis[i]
    
    # Bollinger Band Width percentile (50-period lookback)
    bb_width_percentile = np.full(n, np.nan)
    lookback = 50
    if n >= bb_length + lookback:
        for i in range(bb_length-1 + lookback, n):
            start_idx = i - lookback + 1
            window = bb_width[start_idx:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                current_val = bb_width[i]
                if not np.isnan(current_val):
                    rank = np.sum(valid <= current_val) / len(valid) * 100
                    bb_width_percentile[i] = rank
    
    # Volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = bb_length + lookback + 20  # BB + percentile + vol MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Squeeze condition: low volatility (BB width < 20th percentile)
        squeeze = bb_width_percentile[i] < 20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Bollinger basis OR stoploss
            if (close[i] < basis[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Bollinger basis OR stoploss
            if (close[i] > basis[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: price breaks Bollinger Band with volume during/after squeeze
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: bullish breakout with volume (can occur during or after squeeze)
                if bull_breakout and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with volume (can occur during or after squeeze)
                elif bear_breakout and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals