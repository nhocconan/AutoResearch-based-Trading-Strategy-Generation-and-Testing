#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Uses Camarilla pivot levels (R1/S1) from daily timeframe for breakout entries, filtered by 1-day EMA50 trend and volume confirmation. Designed for low trade frequency (12-37/year) with clear signals in both bull and bear markets by following the daily trend. Uses discrete position sizing (0.25) to minimize fee churn.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Using standard Camarilla formulas: 
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Shift to get previous day's values
    prev_high = np.roll(prev_high, 1)
    prev_low = np.roll(prev_low, 1)
    prev_close = np.roll(prev_close, 1)
    # First day has no previous - set to current values to avoid false signals
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else high[0]
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else low[0]
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else close[0]
    
    # Calculate Camarilla R1 and S1
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily trend filter (EMA50)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + above daily EMA50 + volume
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 + below daily EMA50 + volume
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to Camarilla S1 OR trend turns down
                if (close[i] <= camarilla_s1_aligned[i]) or \
                   (close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to Camarilla R1 OR trend turns up
                if (close[i] >= camarilla_r1_aligned[i]) or \
                   (close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals