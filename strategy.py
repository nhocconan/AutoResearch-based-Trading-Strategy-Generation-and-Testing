#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 Breakout with 12h EMA Trend and Volume Confirmation
# Long when price breaks above S1 with 12h uptrend and volume spike
# Short when price breaks below R1 with 12h downtrend and volume spike
# Uses Camarilla levels from previous day for intraday support/resistance
# Volume filter requires >2x 24-period average volume to confirm institutional interest
# Trend filter uses 12h EMA50 to ensure alignment with higher timeframe momentum
# Designed to capture intraday mean reversion breaks with trend alignment
# Target: 100-180 total trades over 4 years (25-45/year) within optimal range

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
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
    
    # Get daily data for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Camarilla formulas
    R1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    S1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema50_12h[i] = (close_12h[i] * 2 + ema50_12h[i-1] * 48) / 50
    
    # Calculate 24-period average volume for volume filter
    vol_avg_24 = np.full(n, np.nan)
    if n >= 24:
        for i in range(23, n):
            vol_avg_24[i] = np.mean(volume[i-23:i+1])
    
    # Align indicators to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 2x 24-period average
        vol_filter = volume[i] > 2.0 * vol_avg_24[i]
        
        # Determine trend direction from 12h EMA50
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        if position == 0:
            # Look for entry: Camarilla breakout with volume and trend filter
            # Long when price breaks above S1 in uptrend with volume
            long_condition = (
                close[i] > S1_aligned[i] and   # price breaks above S1
                bullish_trend and              # only long in uptrend
                vol_filter                     # volume confirmation
            )
            
            # Short when price breaks below R1 in downtrend with volume
            short_condition = (
                close[i] < R1_aligned[i] and   # price breaks below R1
                bearish_trend and              # only short in downtrend
                vol_filter                     # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below S1 or trend changes
            if close[i] < S1_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above R1 or trend changes
            if close[i] > R1_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals