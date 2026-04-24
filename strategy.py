#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend direction.
- EMA34 > 0 slope indicates bullish trend, EMA34 < 0 slope indicates bearish trend.
- Entry: Long when price breaks above Camarilla R3 AND EMA34 slope > 0 (bullish breakout in uptrend).
         Short when price breaks below Camarilla S3 AND EMA34 slope < 0 (bearish breakout in downtrend).
- Exit: Opposite Camarilla breakout (R4/S4) or EMA slope flips.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h
    ema34 = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_slope = ema34 - np.roll(ema34, 1)
    ema34_slope[0] = 0
    
    # Align 12h EMA34 slope to 4h
    ema34_slope_aligned = align_htf_to_ltf(prices, df_12h, ema34_slope)
    
    # Calculate Camarilla levels from previous 1d (using 1d HTF for pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation
    prev_high = pd.Series(df_1d['high']).shift(1).values
    prev_low = pd.Series(df_1d['low']).shift(1).values
    prev_close = pd.Series(df_1d['close']).shift(1).values
    
    # Calculate Camarilla levels
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 2)  # Need enough 12h bars for EMA and 1d bars for pivots
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_slope_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_slope = ema34_slope_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if ema_slope > 0:  # Bullish trend: look for long breakouts
                    # Bullish breakout: price breaks above Camarilla R3
                    if curr_high > R3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema_slope < 0:  # Bearish trend: look for short breakouts
                    # Bearish breakout: price breaks below Camarilla S3
                    if curr_low < S3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks above R4 (take profit) OR EMA slope flips bearish
            if curr_high > R4_aligned[i] or ema_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks below S4 (take profit) OR EMA slope flips bullish
            if curr_low < S4_aligned[i] or ema_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_12hEMA34Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0