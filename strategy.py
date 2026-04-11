#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with daily ATR-based range and closing price position.
# Uses daily ATR(14) to measure volatility and position within daily range for mean reversion.
# Long when price closes in lower 30% of daily range with high volatility (ATR expansion).
# Short when price closes in upper 30% of daily range with high volatility.
# Volatility filter ensures trades occur during active market conditions.
# Designed for 12-37 trades/year on 12h timeframe with volatility filtering.

name = "12h_1d_atr_range_position_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # ATR(14) using Wilder's smoothing
    atr_14 = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[0:14])  # First ATR is simple average
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily range position (0 = low, 1 = high)
    daily_range = high_1d - low_1d
    range_position = np.full_like(close_1d, np.nan, dtype=float)
    valid_range = daily_range > 0
    range_position[valid_range] = (close_1d[valid_range] - low_1d[valid_range]) / daily_range[valid_range]
    
    # Volatility filter: ATR > 1.5 * ATR(50) for volatility expansion
    atr_50 = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 50:
        for i in range(49, len(tr)):
            atr_50[i] = np.mean(tr[i-49:i+1])
    vol_expansion = atr_14 > (1.5 * atr_50)
    
    # Align daily indicators to 12h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    range_position_aligned = align_htf_to_ltf(prices, df_1d, range_position)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(range_position_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or
            np.isnan(vol_expansion_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        # Long: price in lower 30% of daily range during volatility expansion
        long_entry = (range_position_aligned[i] <= 0.30 and vol_expansion_aligned[i])
        # Short: price in upper 30% of daily range during volatility expansion
        short_entry = (range_position_aligned[i] >= 0.70 and vol_expansion_aligned[i])
        
        # Exit when price returns to middle 40% of daily range (30%-70%)
        exit_long = (position == 1 and range_position_aligned[i] >= 0.30)
        exit_short = (position == -1 and range_position_aligned[i] <= 0.70)
        
        # Generate signals
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals