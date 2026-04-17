#!/usr/bin/env python3
"""
6h_ElderRay_EMA21_RangeFilter_V1
Strategy: Elder Ray (Bull/Bear Power) + EMA21 trend filter + Bollinger Band range filter.
Long: Bull Power > 0, price > EMA21, price below upper Bollinger Band (20,2)
Short: Bear Power < 0, price < EMA21, price above lower Bollinger Band (20,2)
Exit: Price crosses EMA21 in opposite direction
Position size: 0.25
Designed to capture trend continuation in moderate volatility regimes.
Timeframe: 6h
"""

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
    
    # Calculate 6h EMA21
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 6-day EMA13 for Elder Ray (13 periods on 6h = ~3.25 days)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Bollinger Bands (20,2)
    sma20 = close_s.rolling(window=20, min_periods=20).mean().values
    std20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Need EMA21
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to EMA21
        price_above_ema = close[i] > ema21[i]
        price_below_ema = close[i] < ema21[i]
        
        # Range filter: price within Bollinger Bands
        price_below_upper = close[i] < upper_bb[i]
        price_above_lower = close[i] > lower_bb[i]
        
        if position == 0:
            # Long: Bull Power positive + price above EMA21 + below upper BB
            if bull_power[i] > 0 and price_above_ema and price_below_upper:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative + price below EMA21 + above lower BB
            elif bear_power[i] < 0 and price_below_ema and price_above_lower:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA21
            if not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA21
            if not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA21_RangeFilter_V1"
timeframe = "6h"
leverage = 1.0