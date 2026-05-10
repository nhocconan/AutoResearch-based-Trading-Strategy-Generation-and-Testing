#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: On 6h timeframe, trade breakouts of Camarilla R3/S3 levels when weekly trend is aligned and volume spikes.
# Weekly trend: price above/below weekly EMA200 (long-term trend filter).
# Entry: Long when price breaks above R3 with volume > 2x average and weekly uptrend.
# Short when price breaks below S3 with volume > 2x average and weekly downtrend.
# Exit: When price returns to Camarilla R1/S1 levels or volume drops below average.
# Uses Camarilla levels from 1d timeframe for intraday structure.
# Volume spike filters for institutional participation.
# Weekly trend filter ensures we trade with the long-term trend, avoiding counter-trend whipsaws.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from 1d timeframe (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formulas
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Weekly trend filter: EMA200 on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_uptrend = close_1w > ema200_1w
    weekly_downtrend = close_1w < ema200_1w
    
    # Align weekly trend to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volume average (24-period on 6h = 6 days)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + price breaks above R3 + volume spike
            if weekly_up and close[i] > R3_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + price breaks below S3 + volume spike
            elif weekly_down and close[i] < S3_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to R1 or volume drops below average
            if close[i] < R1_aligned[i] or vol_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to S1 or volume drops below average
            if close[i] > S1_aligned[i] or vol_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals