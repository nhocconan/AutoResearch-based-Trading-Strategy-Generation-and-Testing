#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R Extreme Reversal with 1-day Trend Filter and Volume Confirmation.
Williams %R identifies overbought/oversold conditions. In strong trends, these extremes often precede continuations rather than reversals.
We use 1-day EMA50 as trend filter: only take Williams %R reversals in the direction of the daily trend.
Volume spike confirms institutional participation. Designed for low trade frequency by requiring confluence of three filters.
Works in bull markets (buying oversold in uptrend) and bear markets (selling overbought in downtrend).
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1-day close for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average (slightly lower threshold for more signals)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80), above rising 1-day EMA50, volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema50_1d_aligned[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), below falling 1-day EMA50, volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema50_1d_aligned[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses back through opposite extreme or trend fails
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 (momentum fading) or price falls below EMA50
                if williams_r[i] > -50 or close[i] < ema50_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R falls below -50 (momentum fading) or price rises above EMA50
                if williams_r[i] < -50 or close[i] > ema50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Extreme_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0