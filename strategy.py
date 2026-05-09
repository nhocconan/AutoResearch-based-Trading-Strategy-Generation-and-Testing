#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Reversal + 1d Trend + Volume Spike
# Camarilla levels (H3/L3) act as intraday support/resistance in ranging markets.
# Reversal from H3 (short) or L3 (long) with daily trend alignment and volume spike
# captures mean reversion moves with trend filter. Works in both bull/bear via trend filter.
# Target: 20-40 trades/year (80-160 over 4 years) to avoid fee drag.
name = "4h_Camarilla_H3L3_Reversal_1dTrend_Volume"
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
    
    # Get daily data for Camarilla calculation and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC (shifted by 1)
    daily_high = df_daily['high'].shift(1).values
    daily_low = df_daily['low'].shift(1).values
    daily_close = df_daily['close'].shift(1).values
    
    # Pivot point
    pivot = (daily_high + daily_low + daily_close) / 3.0
    # Camarilla levels
    h3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    l3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    h4 = daily_close + (daily_high - daily_low) * 1.1 / 2
    l4 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Align daily Camarilla levels to 4h
    pivot_4h = align_htf_to_ltf(prices, df_daily, pivot)
    h3_4h = align_htf_to_ltf(prices, df_daily, h3)
    l3_4h = align_htf_to_ltf(prices, df_daily, l3)
    h4_4h = align_htf_to_ltf(prices, df_daily, h4)
    l4_4h = align_htf_to_ltf(prices, df_daily, l4)
    
    # Daily EMA50 for trend filter
    ema50_daily = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_4h = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_4h[i]) or np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(h4_4h[i]) or np.isnan(l4_4h[i]) or np.isnan(ema50_daily_4h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Reversal from L3 with daily uptrend and volume spike
            if low[i] <= l3_4h[i] and close[i] > ema50_daily_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Reversal from H3 with daily downtrend and volume spike
            elif high[i] >= h3_4h[i] and close[i] < ema50_daily_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price reaches H4 or daily trend turns down
            if high[i] >= h4_4h[i] or close[i] < ema50_daily_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reaches L4 or daily trend turns up
            if low[i] <= l4_4h[i] or close[i] > ema50_daily_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals