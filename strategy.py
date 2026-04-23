#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d HMA21 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (R3, S3) from 1d for breakout signals
- 1d HMA(21) as trend filter (long only above, short only below) for better trend responsiveness
- Volume > 2.0x 20-period average for confirmation to avoid low-quality breakouts
- Position size: 0.25 discrete level to minimize fee churn
- Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean().values
    # WMA of full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean().values
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R3, S3) from prior 1d bar
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # R3 = pivot + (range * 1.1 / 4)
    # S3 = pivot - (range * 1.1 / 4)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    camarilla_r3 = pivot + (rng * 1.1 / 4.0)
    camarilla_s3 = pivot - (rng * 1.1 / 4.0)
    
    # Align Camarilla levels to 1d timeframe (using completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d data for HMA21 trend filter
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21)  # Volume MA, HMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(hma_21_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_r3_aligned[i]  # Close above R3
        breakout_down = close[i] < camarilla_s3_aligned[i]  # Close below S3
        
        if position == 0:
            # Long: 1d Camarilla R3 breakout up AND price above 1d HMA21 AND volume confirmation
            if breakout_up and close[i] > hma_21_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 1d Camarilla S3 breakout down AND price below 1d HMA21 AND volume confirmation
            elif breakout_down and close[i] < hma_21_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d Camarilla S3 breakdown OR price crosses below 1d HMA21
            if breakout_down or close[i] < hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1d Camarilla R3 breakout OR price crosses above 1d HMA21
            if breakout_up or close[i] > hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dHMA21_VolumeSpike_Filter_v1"
timeframe = "4h"
leverage = 1.0