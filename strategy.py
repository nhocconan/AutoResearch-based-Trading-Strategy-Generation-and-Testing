#!/usr/bin/env python3
"""
4h_1d_Triple_Pivot_Momentum
Hypothesis: Combine 1d Camarilla pivot levels with 4h momentum and volume confirmation to capture institutional breakouts.
Works in bull markets (breakouts above H3/H4) and bear markets (breakdowns below L3/L4). Uses multiple confirmation layers
to reduce false signals and maintain low trade frequency. Target: 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    range_val = high - low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    C = close
    H = high
    L = low
    R1 = C + ((H - L) * 1.0833)
    R2 = C + ((H - L) * 1.1666)
    R3 = C + ((H - L) * 1.2500)
    R4 = C + ((H - L) * 1.5000)
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align all data to 4h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (stricter filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    # 4h momentum filter: price > 20-period EMA for longs, < 20-period EMA for shorts
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    price_above_ema = close > ema_20
    price_below_ema = close < ema_20
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(R4_1d_aligned[i]) or 
            np.isnan(L3_1d_aligned[i]) or np.isnan(L4_1d_aligned[i]) or 
            np.isnan(volume_expansion[i]) or np.isnan(price_above_ema[i]) or 
            np.isnan(price_below_ema[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above R3 with volume expansion and price above EMA
        long_condition = (close[i] > R3_1d_aligned[i]) and volume_expansion[i] and price_above_ema[i]
        
        # Short: breakdown below L3 with volume expansion and price below EMA
        short_condition = (close[i] < L3_1d_aligned[i]) and volume_expansion[i] and price_below_ema[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Triple_Pivot_Momentum"
timeframe = "4h"
leverage = 1.0