#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1-day EMA50 + volume confirmation
# Long when: Alligator aligned bullish (Jaw>Teeth>Lips) AND price > 1-day EMA50 AND volume > 1.5x 20-period average
# Short when: Alligator aligned bearish (Jaw<Teeth<Lips) AND price < 1-day EMA50 AND volume > 1.5x 20-period average
# Exit: price crosses the Alligator Teeth (8-period smoothed median) in opposite direction
# Williams Alligator identifies trend alignment, 1-day EMA50 filters higher timeframe bias, volume confirms momentum
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: SMMA (Smoothed Moving Average) of median price
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMMA of median, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth: 8-period SMMA of median, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips: 5-period SMMA of median, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Calculate EMA50 on 1d
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 13 for Jaw + 8 shift + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get EMA50 values aligned to 4h timeframe
        ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
        ema50_current = ema50_aligned[i]
        
        close_current = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Alligator bullish (Jaw>Teeth>Lips) AND price > EMA50 AND volume confirmation
            if (jaw_val > teeth_val > lips_val and 
                close_current > ema50_current and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Alligator bearish (Jaw<Teeth<Lips) AND price < EMA50 AND volume confirmation
            elif (jaw_val < teeth_val < lips_val and 
                  close_current < ema50_current and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Teeth
            if close_current < teeth_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Teeth
            if close_current > teeth_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0