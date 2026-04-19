#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1_S1_Breakout_With_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (PP, R1, S1)
    # Pivot Point = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    n_days = len(close_1d)
    PP = np.full(n_days, np.nan)
    R1 = np.full(n_days, np.nan)
    S1 = np.full(n_days, np.nan)
    
    for i in range(1, n_days):
        # Use previous day's OHLC to calculate today's pivot
        PP[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        R1[i] = 2 * PP[i] - low_1d[i-1]
        S1[i] = 2 * PP[i] - high_1d[i-1]
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Align pivot levels to 12h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when price breaks above R1 with volume
            if close[i] > R1_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume
            elif close[i] < S1_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below PP (mean reversion to pivot)
            if close[i] < PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above PP (mean reversion to pivot)
            if close[i] > PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals