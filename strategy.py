#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Pullback_Volume
4h strategy using Camarilla pivot levels (R1/S1) with pullback entries and volume confirmation.
- Long: Pullback to S1 in uptrend (HMA21 > HMA50) with volume > 1.5x average
- Short: Pullback to R1 in downtrend (HMA21 < HMA50) with volume > 1.5x average
- Exit: Opposite pullback or trend reversal
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend)
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's OHLC for Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_open = df_1d['open'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1)
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # HMA for trend filter (21 and 50 periods)
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.convolve(arr, np.ones(half)/half, mode='same')
        wma1 = np.convolve(arr, np.ones(period)/period, mode='same')
        raw_hma = 2 * wma2 - wma1
        hma_vals = np.convolve(raw_hma, np.ones(sqrt)/sqrt, mode='same')
        # Set first 'period' values to NaN
        hma_vals[:period] = np.nan
        return hma_vals
    
    hma_21 = hma(close, 21)
    hma_50 = hma(close, 50)
    
    # Volume average (20-period)
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:20] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for HMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = hma_21[i] > hma_50[i]
        downtrend = hma_21[i] < hma_50[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Pullback conditions (price near Camarilla levels)
        pullback_to_s1 = low[i] <= camarilla_s1_aligned[i] * 1.002 and low[i] >= camarilla_s1_aligned[i] * 0.998
        pullback_to_r1 = high[i] >= camarilla_r1_aligned[i] * 0.998 and high[i] <= camarilla_r1_aligned[i] * 1.002
        
        if position == 0:
            # Long: uptrend + volume + pullback to S1
            if uptrend and vol_confirm and pullback_to_s1:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + pullback to R1
            elif downtrend and vol_confirm and pullback_to_r1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or pullback to R1 (opposite level)
            if not uptrend or pullback_to_r1:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or pullback to S1 (opposite level)
            if not downtrend or pullback_to_s1:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Pullback_Volume"
timeframe = "4h"
leverage = 1.0