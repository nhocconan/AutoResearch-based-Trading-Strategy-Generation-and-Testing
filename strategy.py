#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Camarilla R1/S1 breakout + 1w EMA50 trend filter + volume confirmation.
Long when price breaks above Camarilla R1 level with 1w EMA50 uptrend and volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 level with 1w EMA50 downtrend and volume > 1.5x 20-period average.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 12h timeframe to target 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to primary timeframe (12h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 20-period volume average on 12h timeframe
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with 1w EMA50 uptrend and volume
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and  # price above EMA50 = uptrend
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with 1w EMA50 downtrend and volume
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and  # price below EMA50 = downtrend
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla S1 (opposite side)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Camarilla R1 (opposite side)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarilla_R1S1_Volume_Confirm_1wEMA50_Filter"
timeframe = "12h"
leverage = 1.0