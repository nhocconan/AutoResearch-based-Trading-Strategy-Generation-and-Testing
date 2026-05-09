#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Reversal + 1d Trend Filter + Volume Spike
# Uses Camarilla levels for mean reversion in range-bound markets.
# Reversal at S1/R1 with daily trend alignment and volume confirmation.
# Target: 20-30 trades/year (80-120 over 4 years) to avoid fee drag.
# Works in both bull and bear via trend filter and reversal logic.
name = "4h_Camarilla_Reversal_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC (shift 1 to avoid look-ahead)
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    # Camarilla equations
    range_val = prev_high - prev_low
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Resistance levels
    r1 = pivot + (range_val * 1.1 / 12)
    r2 = pivot + (range_val * 1.1 / 6)
    r3 = pivot + (range_val * 1.1 / 4)
    r4 = pivot + (range_val * 1.1 / 2)
    # Support levels
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    
    # Align daily Camarilla levels to 4h
    pivot_4h = align_htf_to_ltf(prices, df_daily, pivot)
    r1_4h = align_htf_to_ltf(prices, df_daily, r1)
    s1_4h = align_htf_to_ltf(prices, df_daily, s1)
    r2_4h = align_htf_to_ltf(prices, df_daily, r2)
    s2_4h = align_htf_to_ltf(prices, df_daily, s2)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(df_daily['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_4h = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or np.isnan(ema34_daily_4h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: Reversal from S1 with daily uptrend and volume spike
            if low[i] <= s1_4h[i] and close[i] > s1_4h[i] and close[i] > ema34_daily_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Reversal from R1 with daily downtrend and volume spike
            elif high[i] >= r1_4h[i] and close[i] < r1_4h[i] and close[i] < ema34_daily_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price reaches R1 or daily trend turns down
            if high[i] >= r1_4h[i] or close[i] < ema34_daily_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reaches S1 or daily trend turns up
            if low[i] <= s1_4h[i] or close[i] > ema34_daily_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals