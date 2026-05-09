#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot (R1/S1) Breakout with 12h EMA50 Trend and Volume Spike
# Uses daily Camarilla pivot levels (R1/S1) from 12h data for breakout signals,
# 12h EMA50 for trend alignment, and volume spike for confirmation.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend with volume).
# Designed for 20-40 trades/year to avoid fee drag.
name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot levels and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Daily Camarilla pivot levels (based on previous day's OHLC)
    # Calculate pivot and levels from 12h data (each 12h bar represents half a day)
    # We need to aggregate to daily equivalent: use 2 consecutive 12h bars for one day
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # For each 12h bar, we'll use the previous 24h (2x12h) range for Camarilla calculation
    # This approximates daily pivot using available 12h data
    if len(close_12h) >= 2:
        # Create daily high/low/close from 2-period rolling window of 12h data
        daily_high = np.maximum(high_12h[:-1], high_12h[1:])  # max of current and previous 12h
        daily_low = np.minimum(low_12h[:-1], low_12h[1:])    # min of current and previous 12h
        daily_close = (close_12h[:-1] + close_12h[1:]) / 2   # average of two 12h closes
        
        # Calculate Camarilla levels for each daily bar
        daily_range = daily_high - daily_low
        pivot = (daily_high + daily_low + daily_close) / 3
        r1 = pivot + 1.1 * daily_range / 12
        s1 = pivot - 1.1 * daily_range / 12
        
        # Now we need to map these back to 12h frequency
        # Each Camarilla level applies to the second 12h bar of the pair
        camarilla_r1_12h = np.full_like(close_12h, np.nan)
        camarilla_s1_12h = np.full_like(close_12h, np.nan)
        
        # For each 12h bar starting from index 1, use the Camarilla levels from the previous day
        camarilla_r1_12h[1:] = r1
        camarilla_s1_12h[1:] = s1
    else:
        camarilla_r1_12h = np.full_like(close_12h, np.nan)
        camarilla_s1_12h = np.full_like(close_12h, np.nan)
    
    # Align Camarilla levels and EMA to 4h
    r1_4h = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    s1_4h = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(ema50_4h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above Camarilla R1 with uptrend and volume spike
            if close[i] > r1_4h[i] and close[i] > ema50_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 with downtrend and volume spike
            elif close[i] < s1_4h[i] and close[i] < ema50_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Camarilla S1 OR trend turns down
            if close[i] < s1_4h[i] or close[i] < ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Camarilla R1 OR trend turns up
            if close[i] > r1_4h[i] or close[i] > ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals