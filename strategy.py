#!/usr/bin/env python3
"""
1d_1w_12h_Camarilla_Breakout
Hypothesis: Breakout above/below weekly Camarilla R4/S4 with 1d trend filter and volume confirmation. Uses weekly support/resistance as structural levels, 1d EMA for trend alignment, and volume spike to confirm institutional participation. Designed for low trade frequency (<15/year) to minimize fee drag while capturing major trend continuations in both bull and bear markets.
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
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for structural levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    R1_1w, R2_1w, R3_1w, R4_1w, S1_1w, S2_1w, S3_1w, S4_1w = calculate_camarilla(high_1w, low_1w, close_1w)
    
    # Align weekly levels to daily
    R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    
    # Get daily data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d trend filter: 50 EMA
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close_1d > ema_50
    downtrend = close_1d < ema_50
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume_1d > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_1w_aligned[i]) or np.isnan(S4_1w_aligned[i]) or 
            np.isnan(uptrend[i]) or np.isnan(downtrend[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: break above weekly R4 in uptrend with volume expansion
        long_condition = (close[i] > R4_1w_aligned[i]) and uptrend[i] and volume_expansion[i]
        
        # Short: break below weekly S4 in downtrend with volume expansion
        short_condition = (close[i] < S4_1w_aligned[i]) and downtrend[i] and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        else:
            # Exit: reverse signal or loss of trend/volume
            if position == 1 and (not uptrend[i] or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not downtrend[i] or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_12h_Camarilla_Breakout"
timeframe = "1d"
leverage = 1.0