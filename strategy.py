#!/usr/bin/env python3
"""
4h_1d_Camarilla_Range_Breakout_Volume
Hypothesis: Price breaking above R4 or below S4 on the 1d chart with volume expansion signals institutional breakout.
Works in bull markets (breakouts above R4) and bear markets (breakdowns below S4). Uses 1d trend filter to avoid false breakouts.
Target: 20-40 trades/year.
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
    
    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d trend filter: price above/below 50 EMA to avoid counter-trend breakouts
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close_1d > ema_50
    downtrend = close_1d < ema_50
    
    # Calculate Camarilla levels on daily
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align all data to 4h timeframe
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: break above R4 in uptrend with volume expansion
        long_condition = (close[i] > R4_1d_aligned[i]) and uptrend_aligned[i] > 0.5 and volume_expansion[i]
        
        # Short: break below S4 in downtrend with volume expansion
        short_condition = (close[i] < S4_1d_aligned[i]) and downtrend_aligned[i] > 0.5 and volume_expansion[i]
        
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
            # Exit conditions: reverse signal or loss of trend/volume
            if position == 1 and (not uptrend_aligned[i] > 0.5 or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not downtrend_aligned[i] > 0.5 or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Camarilla_Range_Breakout_Volume"
timeframe = "4h"
leverage = 1.0