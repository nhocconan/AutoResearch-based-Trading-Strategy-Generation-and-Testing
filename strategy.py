#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with weekly trend filter and volume confirmation.
# Long when green line > red line > blue line (bullish alignment), weekly trend up, volume > 1.5x average.
# Short when green line < red line < blue line (bearish alignment), weekly trend down, volume > 1.5x average.
# Uses discrete position size (0.25) to minimize churn. Designed for 12h timeframe to capture multi-day trends.
# Williams Alligator uses smoothed moving averages (SMMA) of median price: Jaw (13,8), Teeth (8,5), Lips (5,3).
# Weekly trend filter uses EMA50 on weekly data to avoid counter-trend trades.
# Target: 20-40 trades/year per symbol (~80-160 total over 4 years).

name = "12h_WilliamsAlligator_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator lines (using SMMA)
    lips = smma(median_price, 5)   # Green, 5-period, 3-shift
    teeth = smma(median_price, 8)  # Red, 8-period, 5-shift
    jaw = smma(median_price, 13)   # Blue, 13-period, 8-shift
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate EMA50 on weekly
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 12h
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13+8)  # Ensure Alligator lines and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_weekly_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_weekly_aligned[i]
        weekly_downtrend = close[i] < ema_50_weekly_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Enter long if bullish alignment, weekly uptrend, and volume confirmation
            if bullish_alignment and weekly_uptrend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if bearish alignment, weekly downtrend, and volume confirmation
            elif bearish_alignment and weekly_downtrend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when alignment breaks or weekly trend turns down
            if not bullish_alignment or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when alignment breaks or weekly trend turns up
            if not bearish_alignment or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals