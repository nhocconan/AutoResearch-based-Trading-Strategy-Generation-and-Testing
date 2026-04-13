#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout
Hypothesis: Buy breakouts above R3 in uptrend, sell breakdowns below S3 in downtrend using 1w trend filter (price above/below 100-period EMA). Works in bull markets via breakouts and bear markets via breakdowns. Volume confirmation ensures institutional participation. Target: 10-20 trades/year.
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
    R3 = C + ((H - L) * 1.2500)
    S3 = C - ((H - L) * 1.2500)
    return R3, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for breakout levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly trend filter: price above/below 100 EMA
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    uptrend_1w = close_1w > ema_100_1w
    downtrend_1w = close_1w < ema_100_1w
    
    # Calculate Camarilla levels on daily (only R3 and S3)
    R3_1d, S3_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Volume confirmation: current daily volume > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_expansion_1d = volume_1d > (vol_ma_20_1d * 1.5)
    
    # Align all data to 1d timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    volume_expansion_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or
            np.isnan(volume_expansion_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above R3 in uptrend with volume expansion
        long_condition = (high[i] > R3_1d_aligned[i]) and uptrend_1w_aligned[i] > 0.5 and volume_expansion_1d_aligned[i]
        
        # Short: breakdown below S3 in downtrend with volume expansion
        short_condition = (low[i] < S3_1d_aligned[i]) and downtrend_1w_aligned[i] > 0.5 and volume_expansion_1d_aligned[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_Camarilla_Breakout"
timeframe = "1d"
leverage = 1.0