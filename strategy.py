#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (standard period)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13_1d
    bear_power = low - ema13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate EMA13 trend filter from 1d close
    ema13_trend = ema13_1d
    ema13_trend_6h = align_htf_to_ltf(prices, df_1d, ema13_trend)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema13_trend_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if Elder Ray data not ready
        if np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bull power > 0 AND price above EMA13 trend + volume confirmation
            if bull_power_6h[i] > 0 and close[i] > ema13_trend_6h[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bear power < 0 AND price below EMA13 trend + volume confirmation
            elif bear_power_6h[i] < 0 and close[i] < ema13_trend_6h[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bull power <= 0 OR price below EMA13 trend
            if bull_power_6h[i] <= 0 or close[i] < ema13_trend_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bear power >= 0 OR price above EMA13 trend
            if bear_power_6h[i] >= 0 or close[i] > ema13_trend_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals