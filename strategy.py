#!/usr/bin/env python3
"""
1d_Williams_Alligator_MeanReversion_1wTrend_Filter
Hypothesis: On daily timeframe, use Williams Alligator (SMAs of median price) to identify mean reversion opportunities when price deviates significantly from the Alligator's teeth (middle SMA). Filter by weekly trend: only take long deviations in weekly uptrend and short deviations in weekly downtrend. Add volume confirmation to avoid low-liquidity whipsaws. Designed for low trade frequency (10-25/year) with discrete position sizing to minimize fee drag and work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
"""

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
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator: Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    # Using SMMA (Smoothed Moving Average) approximation via EMA with alpha=1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value: simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev_SMMA*(period-1) + Current_Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly trend: 21-period EMA on weekly close
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (longest period 13) + volume mean
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Mean reversion signals: price deviates from Alligator's teeth (middle SMA)
            # Long: price below lips AND teeth in weekly uptrend with volume confirmation
            # Short: price above lips AND teeth in weekly downtrend with volume confirmation
            long_signal = (close[i] < lips[i]) and (close[i] < teeth[i]) and (close_1w[-1] > ema21_1w_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] > lips[i]) and (close[i] > teeth[i]) and (close_1w[-1] < ema21_1w_aligned[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to or above teeth (mean reversion complete)
            exit_signal = close[i] >= teeth[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to or below teeth (mean reversion complete)
            exit_signal = close[i] <= teeth[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Williams_Alligator_MeanReversion_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0