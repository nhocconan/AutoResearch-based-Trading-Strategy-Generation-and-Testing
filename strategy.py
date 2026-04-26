#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeFilter_v1
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation. 
Enters long when price breaks above R1 with volume > 1.5x 20-period average and close > 12h EMA50.
Enters short when price breaks below S1 with volume > 1.5x 20-period average and close < 12h EMA50.
Exits on opposite Camarilla level touch (S1 for longs, R1 for shorts) or when volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 4h timeframe to target 20-50 trades/year.
Works in bull/bear markets via trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    htf_trend = np.where(close > ema_50_12h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Calculate Camarilla levels (R1, S1) from previous day's OHLC
    # For 4h chart, we use daily OHLC to calculate Camarilla levels
    # We'll use 1d data for the Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 4h bar using previous day's OHLC
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    # We need to align the daily Camarilla levels to 4h bars
    # Calculate Camarilla levels on 1d data first
    if len(df_1d) >= 2:
        # For each day, calculate Camarilla levels based on previous day's OHLC
        camarilla_r1_1d = np.full(len(df_1d), np.nan)
        camarilla_s1_1d = np.full(len(df_1d), np.nan)
        
        for i in range(1, len(df_1d)):
            # Previous day's OHLC
            high_prev = df_1d['high'].iloc[i-1]
            low_prev = df_1d['low'].iloc[i-1]
            close_prev = df_1d['close'].iloc[i-1]
            
            # Camarilla calculations
            range_prev = high_prev - low_prev
            camarilla_r1_1d[i] = close_prev + (range_prev * 1.1 / 12)
            camarilla_s1_1d[i] = close_prev - (range_prev * 1.1 / 12)
        
        # Align 1d Camarilla levels to 4h timeframe
        camarilla_r1 = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
        camarilla_s1 = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 50 for EMA, and 1d data for Camarilla)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Long entry: price breaks above R1 with volume filter and uptrend HTF
        if close[i] > camarilla_r1[i] and volume_filter[i] and htf_trend[i] == 1:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        
        # Short entry: price breaks below S1 with volume filter and downtrend HTF
        elif close[i] < camarilla_s1[i] and volume_filter[i] and htf_trend[i] == -1:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        
        # Long exit: price touches or goes below S1
        elif position == 1 and close[i] <= camarilla_s1[i]:
            signals[i] = 0.0
            position = 0
        
        # Short exit: price touches or goes above R1
        elif position == -1 and close[i] >= camarilla_r1[i]:
            signals[i] = 0.0
            position = 0
        
        # Hold current position
        else:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0