#!/usr/bin/env python3
"""
6h_WilliamsAlligator_ElderRay_1wTrend
Hypothesis: Williams Alligator (Jaw, Teeth, Lips) defines market structure, Elder Ray (Bull/Bear Power) measures momentum, 1w EMA200 filters trend direction.
Long when: price above Teeth, Bull Power > 0, and close > 1w EMA200.
Short when: price below Teeth, Bear Power < 0, and close < 1w EMA200.
Uses 6h timeframe for execution with weekly trend filter to avoid whipsaws in ranging markets.
Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

name = "6h_WilliamsAlligator_ElderRay_1wTrend"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (ema_200_1w[i-1] * 199 + close_1w[i]) / 200
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = np.full_like(median_price, np.nan)
    if len(median_price) >= 13:
        for i in range(12, len(median_price)):
            jaw[i] = np.mean(median_price[i-12:i+1])
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = np.full_like(median_price, np.nan)
    if len(median_price) >= 8:
        for i in range(7, len(median_price)):
            teeth[i] = np.mean(median_price[i-7:i+1])
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = np.full_like(median_price, np.nan)
    if len(median_price) >= 5:
        for i in range(4, len(median_price)):
            lips[i] = np.mean(median_price[i-4:i+1])
    lips = np.roll(lips, 3)  # shift 3 bars forward
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = np.full_like(close, np.nan)
    if len(close) >= 13:
        ema13[12] = np.mean(close[0:13])
        for i in range(13, len(close)):
            ema13[i] = (ema13[i-1] * 12 + close[i]) / 13
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(20, 13)  # Ensure Alligator and EMA13 are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(teeth[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price above Teeth, Bull Power positive, and above weekly EMA200
            if (close[i] > teeth[i] and 
                bull_power[i] > 0 and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price below Teeth, Bear Power negative, and below weekly EMA200
            elif (close[i] < teeth[i] and 
                  bear_power[i] < 0 and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: price crosses below Teeth OR trend reversal (below weekly EMA200)
                if close[i] < teeth[i] or close[i] < ema_200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: price crosses above Teeth OR trend reversal (above weekly EMA200)
                if close[i] > teeth[i] or close[i] > ema_200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals