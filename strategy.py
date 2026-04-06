#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price action strategy using 1-week Bollinger Bands breakout with 1-day volume confirmation.
# In bull markets, buy the breakout above upper BB; in bear markets, short the breakdown below lower BB.
# Volume confirms institutional participation; Bollinger Bands adapt to volatility.
# Designed for 12h timeframe to target 50-150 trades over 4 years with low frequency and high win rate.

name = "12h_bb_breakout_1d_vol_1w_v1"
timeframe = "12h"
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
    
    # 1-week Bollinger Bands (20, 2)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands
    bb_length = 20
    bb_mult = 2.0
    
    basis = np.full(len(close_1w), np.nan)
    dev = np.full(len(close_1w), np.nan)
    upper = np.full(len(close_1w), np.nan)
    lower = np.full(len(close_1w), np.nan)
    
    for i in range(bb_length - 1, len(close_1w)):
        basis[i] = np.mean(close_1w[i - bb_length + 1:i + 1])
        dev[i] = bb_mult * np.std(close_1w[i - bb_length + 1:i + 1])
        upper[i] = basis[i] + dev[i]
        lower[i] = basis[i] - dev[i]
    
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    basis_aligned = align_htf_to_ltf(prices, df_1w, basis)
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i - 19:i + 1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(bb_length - 1, 19)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(basis_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below basis or stoploss
            if (close[i] < basis_aligned[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above basis or stoploss
            if (close[i] > basis_aligned[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: breakout above upper Bollinger Band
                if close[i] > upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below lower Bollinger Band
                elif close[i] < lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals