#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with 1-day trend filter and volume spike.
Long when green line > red line (bullish alignment) with price above jaw, 1-day EMA50 rising, and volume spike.
Short when red line > green line (bearish alignment) with price below jaw, 1-day EMA50 falling, and volume spike.
Exit when alignment reverses or price crosses jaw.
Williams Alligator identifies trend presence and direction; 1-day EMA50 filters for higher timeframe trend;
volume spike confirms institutional participation. Designed for low trade frequency by requiring
multiple confirmations. Works in both bull and bear markets by following the 1-day trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator lines (SMMA with specific periods)
    # Jaw: SMMA(13, 8)
    # Teeth: SMMA(8, 5)
    # Lips: SMMA(5, 3)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after enough data for Jaw
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Alligator alignment: bullish when Lips > Teeth > Jaw, bearish when Jaw > Teeth > Lips
        bullish_alignment = lips[i] > teeth[i] > jaw[i]
        bearish_alignment = jaw[i] > teeth[i] > lips[i]
        
        if position == 0:
            # Long: Bullish alignment with price above Jaw, 1-day EMA50 rising, and volume spike
            if (bullish_alignment and 
                close[i] > jaw[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment with price below Jaw, 1-day EMA50 falling, and volume spike
            elif (bearish_alignment and 
                  close[i] < jaw[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alignment reverses or price crosses Jaw
            exit_signal = False
            
            if position == 1:
                # Exit long: Bearish alignment or price crosses below Jaw
                if bearish_alignment or close[i] < jaw[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bullish alignment or price crosses above Jaw
                if bullish_alignment or close[i] > jaw[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Williams_Alligator_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0