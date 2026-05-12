#!/usr/bin/env python3
name = "6h_ElderRay_Alligator_Trend_Signal_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Elder Ray components (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate Alligator lines (13, 8, 5 SMMA)
    def smma(series, period):
        sma = np.full_like(series, np.nan, dtype=float)
        sma[period-1] = np.mean(series[:period])
        for i in range(period, len(series)):
            sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Lips > Teeth > Jaw (bullish alignment), volume filter
            if bull_power[i] > 0 and lips[i] > teeth[i] > jaw[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Jaws > Teeth > Lips (bearish alignment), volume filter
            elif bear_power[i] < 0 and jaw[i] > teeth[i] > lips[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or bearish alignment (Jaw > Teeth)
            if bull_power[i] <= 0 or jaw[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or bullish alignment (Lips > Teeth)
            if bear_power[i] >= 0 or lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals