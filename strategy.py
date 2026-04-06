#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversion with 1d Trend Filter and Volume Confirmation
Hypothesis: Camarilla pivot levels (S3/S4 for long, R3/R4 for short) act as strong support/resistance.
In up-trend (1d EMA200 rising), buy at S3/S4 with volume confirmation.
In down-trend (1d EMA200 falling), sell at R3/R4 with volume confirmation.
This mean-reversion strategy works in both bull (buy dips) and bear (sell rallies).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_prev = np.roll(ema200_1d, 1)
    ema200_prev[0] = ema200_1d[0]
    ema200_rising = ema200_1d > ema200_prev
    ema200_falling = ema200_1d < ema200_prev
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    ema200_rising_aligned = align_htf_to_ltf(prices, df_1d, ema200_rising)
    ema200_falling_aligned = align_htf_to_ltf(prices, df_1d, ema200_falling)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Camarilla: Range = high - low
    # S3 = close - (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1
    # R3 = close + (high - low) * 1.1/2
    # R4 = close + (high - low) * 1.1
    # Note: Using previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_val = prev_high - prev_low
    s3 = prev_close - range_val * 1.1 / 2
    s4 = prev_close - range_val * 1.1
    r3 = prev_close + range_val * 1.1 / 2
    r4 = prev_close + range_val * 1.1
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(ema200_rising_aligned[i]) or 
            np.isnan(ema200_falling_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: mean reversion target or stoploss
        if position == 1:  # long position
            # Exit: price reaches midpoint (Camarilla Q3) or stoploss
            midpoint = (s3[i] + s4[i]) / 2 + (r3[i] + r4[i]) / 2  # Actually just (s3+s4+r3+r4)/4
            midpoint = (s3[i] + s4[i] + r3[i] + r4[i]) / 4
            if (close[i] >= midpoint or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches midpoint or stoploss
            midpoint = (s3[i] + s4[i] + r3[i] + r4[i]) / 4
            if (close[i] <= midpoint or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla S3/S4 for long, R3/R4 for short + trend + volume
            long_entry = ((close[i] <= s3[i] * 1.001 or close[i] <= s4[i] * 1.001) and  # Allow small slippage
                         ema200_rising_aligned[i] and 
                         volume[i] > vol_ema[i] * 1.5)
            short_entry = ((close[i] >= r3[i] * 0.999 or close[i] >= r4[i] * 0.999) and  # Allow small slippage
                          ema200_falling_aligned[i] and 
                          volume[i] > vol_ema[i] * 1.5)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals