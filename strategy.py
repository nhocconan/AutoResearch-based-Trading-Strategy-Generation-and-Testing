#!/usr/bin/env python3
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
    
    # Calculate daily Williams Alligator components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMMA
    close_1d = df_1d['close'].values
    # SMMA (Smoothed Moving Average) calculation
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)  # Blue line
    teeth = smma(close_1d, 8)  # Red line
    lips = smma(close_1d, 5)   # Green line
    
    # Williams Alligator signals: 
    # Bullish: Lips > Teeth > Jaw (green > red > blue)
    # Bearish: Jaw > Teeth > Lips (blue > red > green)
    bullish_alligator = (lips > teeth) & (teeth > jaw)
    bearish_alligator = (jaw > teeth) & (teeth > lips)
    
    # Calculate 6-period ATR for volatility filter
    def calculate_atr(high_arr, low_arr, close_arr, period):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = np.zeros_like(tr)
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_6 = calculate_atr(high, low, close, 6)
    
    # Align higher timeframe data to 6m
    bullish_alligator_aligned = align_htf_to_ltf(prices, df_1d, bullish_alligator)
    bearish_alligator_aligned = align_htf_to_ltf(prices, df_1d, bearish_alligator)
    atr_6_aligned = align_htf_to_ltf(prices, df_1d, atr_6)
    
    # Volume confirmation: current volume > 1.8x 6-period average
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_surge = volume > (vol_ma_6 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_alligator_aligned[i]) or np.isnan(bearish_alligator_aligned[i]) or 
            np.isnan(atr_6_aligned[i]) or np.isnan(volume_surge[i]) or atr_6_aligned[i] == 0):
            signals[i] = 0.0
            continue
        
        # Williams Alligator signals
        bullish = bullish_alligator_aligned[i]
        bearish = bearish_alligator_aligned[i]
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_6_aligned[i] > np.nanmedian(atr_6_aligned[max(0, i-50):i+1])
        
        # Entry conditions
        # Long: Bullish Alligator alignment + volume surge + volatility filter
        long_entry = bullish and volume_surge[i] and vol_filter
        # Short: Bearish Alligator alignment + volume surge + volatility filter
        short_entry = bearish and volume_surge[i] and vol_filter
        
        # Exit conditions: opposing Alligator signal or loss of volume surge
        long_exit = bearish or not volume_surge[i]
        short_exit = bullish or not volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_BullBear_Volume"
timeframe = "6h"
leverage = 1.0