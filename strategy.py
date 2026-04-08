#!/usr/bin/env python3
"""
12h Camarilla Pivot + Weekly Trend + Volume Confirmation v1
Hypothesis: Camarilla pivot levels from weekly timeframe provide strong support/resistance.
Trades breakouts above/below pivot levels with weekly EMA trend alignment and volume confirmation.
Designed for 12h timeframe to capture swing moves in both bull and bear markets with low trade frequency.
Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_weekly_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivots and EMA
    df_1w = get_htf_data(prices, '1w')
    
    # Camarilla pivot levels (based on previous week)
    # Typical price = (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pivot = typical_price.rolling(window=1, min_periods=1).mean()  # Current week's typical price
    high_week = df_1w['high'].rolling(window=1, min_periods=1).max()
    low_week = df_1w['low'].rolling(window=1, min_periods=1).min()
    
    # Calculate Camarilla levels using previous week's data
    # Shift by 1 to use only completed weekly data
    prev_high = high_week.shift(1)
    prev_low = low_week.shift(1)
    prev_close = df_1w['close'].shift(1)
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    camarilla_h5 = prev_close + (range_val * 1.1 / 2)  # Resistance level
    camarilla_l5 = prev_close - (range_val * 1.1 / 2)  # Support level
    camarilla_h4 = prev_close + (range_val * 1.1 / 4)
    camarilla_l4 = prev_close - (range_val * 1.1 / 4)
    camarilla_h3 = prev_close + (range_val * 1.1 / 6)
    camarilla_l3 = prev_close - (range_val * 1.1 / 6)
    
    # Weekly EMA(21) for trend filter
    ema_21 = df_1w['close'].ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Align to 12h timeframe
    camarilla_h5_12h = align_htf_to_ltf(prices, df_1w, camarilla_h5.values)
    camarilla_l5_12h = align_htf_to_ltf(prices, df_1w, camarilla_l5.values)
    camarilla_h4_12h = align_htf_to_ltf(prices, df_1w, camarilla_h4.values)
    camarilla_l4_12h = align_htf_to_ltf(prices, df_1w, camarilla_l4.values)
    camarilla_h3_12h = align_htf_to_ltf(prices, df_1w, camarilla_h3.values)
    camarilla_l3_12h = align_htf_to_ltf(prices, df_1w, camarilla_l3.values)
    ema_21_12h = align_htf_to_ltf(prices, df_1w, ema_21.values)
    
    # Volume filter (>1.5x 20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h5_12h[i]) or np.isnan(camarilla_l5_12h[i]) or 
            np.isnan(ema_21_12h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 or trend reverses
            if close[i] <= camarilla_l3_12h[i] or close[i] < ema_21_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 or trend reverses
            if close[i] >= camarilla_h3_12h[i] or close[i] > ema_21_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long at Camarilla H5 with trend alignment
            if (close[i] >= camarilla_h5_12h[i] and 
                close[i] > ema_21_12h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at Camarilla L5 with trend alignment
            elif (close[i] <= camarilla_l5_12h[i] and 
                  close[i] < ema_21_12h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals