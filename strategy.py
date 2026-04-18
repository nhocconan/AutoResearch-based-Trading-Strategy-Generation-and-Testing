#!/usr/bin/env python3
"""
6h_WeeklyPivot_Reversal
Weekly pivot reversal strategy for 6h timeframe:
- Long when price breaks above weekly R1 with volume confirmation and RSI < 50 (mean reversion in uptrend)
- Short when price breaks below weekly S1 with volume confirmation and RSI > 50 (mean reversion in downtrend)
- Exit when price returns to weekly pivot (PP) or opposite reversal signal occurs
- Uses weekly pivot points from higher timeframe for institutional levels
- Designed for 10-25 trades/year per symbol with strict entry conditions
Works in both bull (buy dips to support) and bear (sell rallies to resistance) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: PP, R1, S1, R2, S2."""
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    return pp, r1, s1, r2, s2

def calculate_rsi(close, period=14):
    """Calculate RSI with Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing: first average is simple average
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    # Subsequent values: smoothed average
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    pp_weekly, r1_weekly, s1_weekly, r2_weekly, s2_weekly = calculate_weekly_pivot(
        weekly_high, weekly_low, weekly_close
    )
    
    # Align weekly pivot points to 6h timeframe
    pp_weekly_6h = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    r1_weekly_6h = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_6h = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Calculate RSI on 6h
    rsi_6h = calculate_rsi(close, 14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # need weekly data and RSI
    
    for i in range(start_idx, n):
        # Skip if weekly pivot data not available
        if (np.isnan(pp_weekly_6h[i]) or np.isnan(r1_weekly_6h[i]) or 
            np.isnan(s1_weekly_6h[i]) or np.isnan(rsi_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume_filter[i] if i < len(volume_filter) else False
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume and RSI < 50 (not overbought)
            if (close[i] > r1_weekly_6h[i] and 
                close[i-1] <= r1_weekly_6h[i-1] and 
                vol_ok and 
                rsi_6h[i] < 50):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume and RSI > 50 (not oversold)
            elif (close[i] < s1_weekly_6h[i] and 
                  close[i-1] >= s1_weekly_6h[i-1] and 
                  vol_ok and 
                  rsi_6h[i] > 50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly pivot or breaks below S1
            if (close[i] <= pp_weekly_6h[i] or 
                (close[i] < s1_weekly_6h[i] and close[i-1] >= s1_weekly_6h[i-1])):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot or breaks above R1
            if (close[i] >= pp_weekly_6h[i] or 
                (close[i] > r1_weekly_6h[i] and close[i-1] <= r1_weekly_6h[i-1])):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Reversal"
timeframe = "6h"
leverage = 1.0