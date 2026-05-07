#!/usr/bin/env python3
name = "6h_PostRangeBreakout_1dTrend_Volume"
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
    
    # Get 1d data for trend filter and range detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 14-day ATR for range detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate Bollinger Bands (20, 2) on 1d
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate volume confirmation (current volume vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Define range as Bollinger Band width
        bb_width = upper_bb_aligned[i] - lower_bb_aligned[i]
        range_threshold = 0.5 * bb_width  # Consider in range if within half the BB width
        
        # Check if price is within the 1d range (between BBands)
        in_range = (close[i] >= lower_bb_aligned[i] - range_threshold) and (close[i] <= upper_bb_aligned[i] + range_threshold)
        
        if position == 0:
            # Look for breakout after ranging period
            if in_range:
                # Bullish breakout: price closes above upper BB with volume and uptrend
                if (close[i] > upper_bb_aligned[i] and 
                    close[i] > ema_21_1d_aligned[i] and 
                    volume_ratio[i] > 2.0):
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below lower BB with volume and downtrend
                elif (close[i] < lower_bb_aligned[i] and 
                      close[i] < ema_21_1d_aligned[i] and 
                      volume_ratio[i] > 2.0):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price closes below EMA21 (trend change)
            if close[i] < ema_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above EMA21 (trend change)
            if close[i] > ema_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals