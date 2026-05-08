#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation.
# Long when price breaks above R3 AND 1d EMA34 rising AND volume > 2.0x 20-period average.
# Short when price breaks below S3 AND 1d EMA34 falling AND volume > 2.0x 20-period average.
# Exit when price crosses back to H4/L4 levels (mean reversion zone).
# Camarilla levels from 1d provide institutional support/resistance. 1d EMA34 ensures trend alignment.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3_S3_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    typical = (high + low + close) / 3
    range_val = high - low
    H4 = typical + (range_val * 1.1 / 2)
    H3 = typical + (range_val * 1.1 / 4)
    H2 = typical + (range_val * 1.1 / 6)
    H1 = typical + (range_val * 1.1 / 12)
    L1 = typical - (range_val * 1.1 / 12)
    L2 = typical - (range_val * 1.1 / 6)
    L3 = typical - (range_val * 1.1 / 4)
    L4 = typical - (range_val * 1.1 / 2)
    return H3, L3, H4, L4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    H3_1d, L3_1d, H4_1d, L4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 12h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or 
            np.isnan(H4_1d_aligned[i]) or np.isnan(L4_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above H3, 1d EMA34 rising, volume filter
            long_cond = (close[i] > H3_1d_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below L3, 1d EMA34 falling, volume filter
            short_cond = (close[i] < L3_1d_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back to H4 (mean reversion zone)
            if close[i] < H4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back to L4 (mean reversion zone)
            if close[i] > L4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals