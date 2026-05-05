#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter (HMA21) and volume confirmation
# Long when: price breaks above Camarilla R3 AND 1d HMA21 is rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Camarilla S3 AND 1d HMA21 is falling AND volume > 1.5x 20-period MA
# Exit when: price returns to Camarilla Pivot point (PP) OR trend reverses
# Uses Camarilla levels for intraday support/resistance, 1d HMA for trend filter, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 80-180 total trades over 4 years (20-45/year) to balance edge and fees.

name = "6h_Camarilla_R3S3_Breakout_1dHMA21_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 6h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Camarilla levels on 6h using previous bar's OHLC
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_pp = np.zeros(n)
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    
    for i in range(1, n):  # Start from 1 to use previous bar
        camarilla_pp[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
        camarilla_r3[i] = close[i-1] + (high[i-1] - low[i-1]) * 1.1 / 2.0
        camarilla_s3[i] = close[i-1] - (high[i-1] - low[i-1]) * 1.1 / 2.0
    
    # For first bar, set to close price to avoid division issues
    camarilla_pp[0] = close[0]
    camarilla_r3[0] = close[0]
    camarilla_s3[0] = close[0]
    
    # Breakout conditions
    breakout_above_r3 = close > camarilla_r3
    breakout_below_s3 = close < camarilla_s3
    return_to_pp = np.abs(close - camarilla_pp) < 0.001 * close  # Within 0.1% of PP
    
    # Get 1d data ONCE before loop for HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 21-period HMA on 1d timeframe
    if len(close_1d) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # Calculate WMA for full period
        wma_21 = np.full(len(close_1d), np.nan)
        for i in range(20, len(close_1d)):
            wma_21[i] = np.mean(close_1d[i-20:i+1] * np.arange(1, 22))
        
        # Calculate WMA for half period
        wma_half = np.full(len(close_1d), np.nan)
        for i in range(half_n-1, len(close_1d)):
            wma_half[i] = np.mean(close_1d[i-half_n+1:i+1] * np.arange(1, half_n+1))
        
        # HMA calculation
        hma_21 = np.full(len(close_1d), np.nan)
        for i in range(20, len(close_1d)):
            if not np.isnan(wma_21[i]) and not np.isnan(wma_half[i]):
                hma_21[i] = 2 * wma_half[i] - wma_21[i]
        
        # Final WMA of the difference
        hma_final = np.full(len(close_1d), np.nan)
        for i in range(sqrt_n-1, len(close_1d)):
            if not np.isnan(hma_21[i-sqrt_n+1:i+1]).all():
                hma_final[i] = np.mean(hma_21[i-sqrt_n+1:i+1] * np.arange(1, sqrt_n+1))
        
        # Trend direction
        hma_rising = np.diff(hma_final, prepend=np.nan) > 0
        hma_falling = np.diff(hma_final, prepend=np.nan) < 0
    else:
        hma_rising = np.full(len(close_1d), False)
        hma_falling = np.full(len(close_1d), False)
    
    # Align 1d HMA trend to 6h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_pp[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above R3 + 1d HMA rising + volume filter
            if (breakout_above_r3[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below S3 + 1d HMA falling + volume filter
            elif (breakout_below_s3[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to PP OR 1d HMA turns falling
            if (return_to_pp[i] or hma_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to PP OR 1d HMA turns rising
            if (return_to_pp[i] or hma_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals