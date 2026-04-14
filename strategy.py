#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with 1-day trend filter (EMA13) and volume confirmation
# Long when price > Alligator teeth (green line) AND price > daily EMA13 AND volume > 1.5x 20-period average
# Short when price < Alligator teeth AND price < daily EMA13 AND volume > 1.5x 20-period average
# Exit when price crosses back below/above the Alligator teeth
# Williams Alligator uses smoothed moving averages (SMA) with specific periods: Jaw(13,8), Teeth(8,5), Lips(5,3)
# This strategy captures strong trending moves with Williams Alligator as trend filter, avoiding counter-trend trades
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for EMA13 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components on 4h
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Calculate daily EMA13 for trend filter
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (13 for Alligator + buffer)
    start = 25
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(teeth[i]) or np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price above Alligator teeth + above daily EMA13 + volume confirmation
            if (price > teeth[i] and price > ema13_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price below Alligator teeth + below daily EMA13 + volume confirmation
            elif (price < teeth[i] and price < ema13_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below Alligator teeth
            if price < teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Alligator teeth
            if price > teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsAlligator_1dEMA13_Volume"
timeframe = "4h"
leverage = 1.0