#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams Alligator + 1w HMA trend filter + volume confirmation
    # Williams Alligator identifies trend absence/presence via smoothed medians (Jaw/Teeth/Lips)
    # 1w HMA21 filters for weekly trend alignment to avoid counter-trend whipsaws
    # Volume spike >1.8x 20-period average confirms institutional participation
    # Target: 15-25 trades/year (60-100 total over 4 years) for minimal fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator on 1d (Smoothed Medians)
    # Jaw: 13-period SMMA of median, shifted 8 bars
    # Teeth: 8-period SMMA of median, shifted 5 bars
    # Lips: 5-period SMMA of median, shifted 3 bars
    median_1d = (high_1d + low_1d) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current) / Period
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1d = smma(median_1d, 13)
    teeth_1d = smma(median_1d, 8)
    lips_1d = smma(median_1d, 5)
    
    # Shift as per Alligator definition
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Get 1w HMA21 for trend filter
    def hma(arr, period):
        """Hull Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma_2x = np.full_like(arr, np.nan)
        wma_n = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= half - 1:
                wma_2x[i] = np.nansum(arr[i-half+1:i+1] * np.arange(1, half+1)) / (half * (half + 1) / 2)
            if i >= period - 1:
                wma_n[i] = np.nansum(arr[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
        raw_hma = 2 * wma_2x - wma_n
        result = np.full_like(arr, np.nan)
        for i in range(sqrt - 1, len(arr)):
            if i >= sqrt - 1:
                result[i] = np.nansum(raw_hma[i-sqrt+1:i+1] * np.arange(1, sqrt+1)) / (sqrt * (sqrt + 1) / 2)
        return result
    
    hma21_1w = hma(close_1w, 21)
    
    # Get 1d volume for confirmation (>1.8x 20-period average)
    vol_ma_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_1d[i] = np.mean(volume[i-20:i])
    volume_spike_1d = volume > (1.8 * vol_ma_1d)
    
    # Align all indicators to LTF (1d)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    hma21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(hma21_1w_aligned[i]) or 
            np.isnan(volume_spike_1d[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions: Lips > Teeth > Jaw = bullish | Lips < Teeth < Jaw = bearish
        bullish_alligator = lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]
        bearish_alligator = lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]
        
        # 1w trend filter (HMA21)
        bullish_trend = close[i] > hma21_1w_aligned[i]
        bearish_trend = close[i] < hma21_1w_aligned[i]
        
        # Entry logic: Alligator alignment + trend alignment + volume confirmation
        long_entry = bullish_alligator and bullish_trend and volume_spike_1d[i]
        short_entry = bearish_alligator and bearish_trend and volume_spike_1d[i]
        
        # Exit logic: Alligator reversal or trend change
        long_exit = not bullish_alligator or not bullish_trend
        short_exit = not bearish_alligator or not bearish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_williams_alligator_hma21_volume_v1"
timeframe = "1d"
leverage = 1.0