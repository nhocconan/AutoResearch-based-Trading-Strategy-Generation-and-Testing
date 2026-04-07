#!/usr/bin/env python3
"""
1d_weekly_pivot_breakout_volume_v1
Hypothesis: Buy when price breaks above weekly pivot resistance with volume confirmation in an uptrend,
sell when price breaks below weekly pivot support with volume confirmation in a downtrend.
Uses weekly pivot levels as key support/resistance, volume to confirm breakout strength,
and daily EMA for trend filter. Designed for low-frequency, high-conviction trades.
Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_breakout_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for pivot levels and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week)
    # Pivot = (H + L + C) / 3
    # Resistance 1 = (2 * Pivot) - L
    # Support 1 = (2 * Pivot) - H
    weekly_high = df_weekly['high'].shift(1).values
    weekly_low = df_weekly['low'].shift(1).values
    weekly_close = df_weekly['close'].shift(1).values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = (2 * pivot) - weekly_low
    s1 = (2 * pivot) - weekly_high
    
    # Weekly EMA for trend filter (20-period)
    weekly_ema = df_weekly['close'].ewm(span=20, adjust=False).mean().values
    
    # Align all weekly data to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(weekly_ema_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly pivot or trend turns bearish
            if close[i] < pivot_aligned[i] or close[i] < weekly_ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above weekly pivot or trend turns bullish
            if close[i] > pivot_aligned[i] or close[i] > weekly_ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R1 with volume and bullish trend
            if (close[i] > r1_aligned[i] and vol_confirm and 
                close[i] > weekly_ema_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S1 with volume and bearish trend
            elif (close[i] < s1_aligned[i] and vol_confirm and 
                  close[i] < weekly_ema_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals