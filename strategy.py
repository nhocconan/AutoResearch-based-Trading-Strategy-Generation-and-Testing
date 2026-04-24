#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend direction.
- Camarilla levels calculated from prior 1d (daily) high/low/close.
- Entry: Long when price breaks above Camarilla R3 AND 12h EMA50 > EMA200 (bullish trend).
         Short when price breaks below Camarilla S3 AND 12h EMA50 < EMA200 (bearish trend).
- Exit: Opposite Camarilla level break (R4/S4) or EMA trend flip.
- Volume confirmation: current 6h volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (prior day's HLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d
    # HLC from previous daily bar
    prev_high = df_1d['high'].shift(1).values  # shift to use prior day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # True range for Camarilla
    true_range = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h5 = prev_close + 1.1 * true_range / 2  # R4 equivalent
    camarilla_h4 = prev_close + 1.1 * true_range / 4  # R3
    camarilla_h3 = prev_close + 1.1 * true_range / 6  # R2
    camarilla_l3 = prev_close - 1.1 * true_range / 6  # S2
    camarilla_l4 = prev_close - 1.1 * true_range / 4  # S3
    camarilla_l5 = prev_close - 1.1 * true_range / 2  # S4
    
    # Align Camarilla levels to 6h (use prior day's levels for current 6h bars)
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # Calculate EMAs on 12h
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMAs to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 200, 20)  # Need EMA200 ready and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: bullish if EMA50 > EMA200, bearish if EMA50 < EMA200
        bullish_trend = ema_50_aligned[i] > ema_200_aligned[i]
        bearish_trend = ema_50_aligned[i] < ema_200_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Bullish breakout: price breaks above H4 (R3) in bullish trend
                if bullish_trend and curr_high > h4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L4 (S3) in bearish trend
                elif bearish_trend and curr_low < l4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks above H5 (R4) or trend turns bearish
            if curr_high > h5_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks below L5 (S4) or trend turns bullish
            if curr_low < l5_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_12hEMATrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0