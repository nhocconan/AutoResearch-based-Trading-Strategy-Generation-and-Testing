#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot Reversal with 1-day Volume Filter and Chop Filter
Long when price crosses below Camarilla S1 (strong support) in low volatility regime with high volume.
Short when price crosses above Camarilla R1 (strong resistance) in low volatility regime with high volume.
Exit when price reaches opposite Camarilla level (S3/R3) or volatility increases.
Uses institutional pivot levels with volume confirmation and volatility filter to avoid choppy losses.
Works in both bull and bear markets by fading extremes at institutional levels during low volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day - need OHLC from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values  # Previous day close
    prev_high = df_1d['high'].shift(1).values    # Previous day high
    prev_low = df_1d['low'].shift(1).values      # Previous day low
    
    # Calculate Camarilla levels for previous day
    range_ = prev_high - prev_low
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H1 = close + (range * 1.1/12), H2 = close + (range * 1.1/6), H3 = close + (range * 1.1/4), H4 = close + (range * 1.1/2)
    # L1 = close - (range * 1.1/12), L2 = close - (range * 1.1/6), L3 = close - (range * 1.1/4), L4 = close - (range * 1.1/2)
    H1 = prev_close + (range_ * 1.1 / 12)
    H2 = prev_close + (range_ * 1.1 / 6)
    H3 = prev_close + (range_ * 1.1 / 4)
    H4 = prev_close + (range_ * 1.1 / 2)
    L1 = prev_close - (range_ * 1.1 / 12)
    L2 = prev_close - (range_ * 1.1 / 6)
    L3 = prev_close - (range_ * 1.1 / 4)
    L4 = prev_close - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (they change only at daily boundaries)
    H1_4h = align_htf_to_ltf(prices, df_1d, H1)
    H2_4h = align_htf_to_ltf(prices, df_1d, H2)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    L1_4h = align_htf_to_ltf(prices, df_1d, L1)
    L2_4h = align_htf_to_ltf(prices, df_1d, L2)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1-day volume filter
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Chopiness index filter (using 1d data for regime detection)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for chop calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # First value
    
    # Chopiness index: log(sum(tr,14)) / (log(14) * true_range) * 100
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h price data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H1_4h[i]) or np.isnan(L1_4h[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade when chop > 50 (ranging market)
        in_range = chop_aligned[i] > 50
        
        if position == 0:
            # Long: Price crosses below L1 (strong support) in ranging market with volume confirmation
            if (in_range and 
                close_4h[i] <= L1_4h[i] and 
                close_4h[i-1] > L1_4h[i-1] and  # Crossed below
                volume_1d[i] > avg_vol_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses above H1 (strong resistance) in ranging market with volume confirmation
            elif (in_range and 
                  close_4h[i] >= H1_4h[i] and 
                  close_4h[i-1] < H1_4h[i-1] and  # Crossed above
                  volume_1d[i] > avg_vol_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit when price reaches H3 (resistance) or volatility increases (chop < 40)
                if (close_4h[i] >= H3_4h[i] or chop_aligned[i] < 40):
                    exit_signal = True
            else:  # position == -1, Short position
                # Exit when price reaches L3 (support) or volatility increases (chop < 40)
                if (close_4h[i] <= L3_4h[i] or chop_aligned[i] < 40):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_Pivot_Reversal_Volume_Chop_Filter"
timeframe = "4h"
leverage = 1.0