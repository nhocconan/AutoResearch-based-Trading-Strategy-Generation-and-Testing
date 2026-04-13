#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Pivot_Trend
Hypothesis: Buy pullbacks to S3 in uptrend, sell rallies to R3 in downtrend using 1d trend filter (price above/below 200-period EMA).
Adds 1w trend filter for stronger confirmation: require both 1d and 1w trend aligned.
Works in bull markets via pullbacks to support and bear markets via rallies to resistance.
Volume confirmation ensures institutional participation. Target: 15-25 trades/year.
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
    
    # Get daily and weekly data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1d trend filter: price above/below 200 EMA
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    uptrend_1d = close_1d > ema_200_1d
    downtrend_1d = close_1d < ema_200_1d
    
    # 1w trend filter: price above/below 200 EMA (approx 40 weeks)
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    uptrend_1w = close_1w > ema_200_1w
    downtrend_1w = close_1w < ema_200_1w
    
    # Calculate Camarilla levels on daily
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align all data to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: pullback to S3 in uptrend (both 1d and 1w) with volume expansion
        long_condition = (low[i] <= S3_1d_aligned[i]) and uptrend_1d_aligned[i] > 0.5 and uptrend_1w_aligned[i] > 0.5 and volume_expansion[i]
        
        # Short: rally to R3 in downtrend (both 1d and 1w) with volume expansion
        short_condition = (high[i] >= R3_1d_aligned[i]) and downtrend_1d_aligned[i] > 0.5 and downtrend_1w_aligned[i] > 0.5 and volume_expansion[i]
        
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
            if position == 1 and (not (uptrend_1d_aligned[i] > 0.5 and uptrend_1w_aligned[i] > 0.5) or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not (downtrend_1d_aligned[i] > 0.5 and downtrend_1w_aligned[i] > 0.5) or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_1w_Camarilla_Pivot_Trend"
timeframe = "12h"
leverage = 1.0