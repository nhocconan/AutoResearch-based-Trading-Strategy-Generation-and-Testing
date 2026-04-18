#!/usr/bin/env python3
"""
12h Williams Alligator with Weekly Trend Filter and Volume Confirmation
Long: Jaw < Teeth < Lips (bullish alignment) + price above Lips + weekly close > weekly open + volume > 1.5x average
Short: Jaw > Teeth > Lips (bearish alignment) + price below Lips + weekly close < weekly open + volume > 1.5x average
Exit: Opposite alignment or price crosses Teeth
Designed to capture trends in both bull and bear markets with strict entry conditions to limit trades.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with future shifts"""
    # Jaw: 13-period SMMA shifted 8 bars
    sma13 = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = sma13.shift(8)
    
    # Teeth: 8-period SMMA shifted 5 bars
    sma8 = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = sma8.shift(5)
    
    # Lips: 5-period SMMA shifted 3 bars
    sma5 = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = sma5.shift(3)
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 21:  # need enough for Lip calculation (5+3 shift)
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Alligator on 12h
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Weekly bullish/bearish close
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align weekly data to 12h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 21  # need Alligator calculations (13+8 shift)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(avg_volume[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: Bullish alignment + price above Lips + weekly bullish + volume
            if jaw[i] < teeth[i] and teeth[i] < lips[i] and price > lips[i] and \
               weekly_bullish_aligned[i] > 0.5 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price below Lips + weekly bearish + volume
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and price < lips[i] and \
                 weekly_bearish_aligned[i] > 0.5 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment OR price crosses below Teeth
            if jaw[i] > teeth[i] or price < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment OR price crosses above Teeth
            if jaw[i] < teeth[i] or price > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0