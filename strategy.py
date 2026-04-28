#!/usr/bin/env python3
"""
6h_ElderRay_BullPower_BearPower_1dTrend_Volume
Hypothesis: Uses Elder Ray (Bull Power/Bear Power) with 1d trend filter and volume spike to capture high-probability momentum moves. Elder Ray measures bullish/bearish power relative to EMA13. Works in both bull and bear by following 1d trend. Targets 15-25 trades/year via strict conditions.
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
    
    # Get 1d data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (1d)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = ema13_1d - df_1d['low'].values
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: >2x 20-period MA (5 days of 6h bars)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Elder Ray conditions: strong bullish/bearish power
        long_signal = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and vol_confirm and uptrend
        short_signal = bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and vol_confirm and downtrend
        
        # Exit conditions: power weakening or opposite signal
        long_exit = bull_power_aligned[i] <= 0
        short_exit = bear_power_aligned[i] <= 0
        
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_BullPower_BearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0