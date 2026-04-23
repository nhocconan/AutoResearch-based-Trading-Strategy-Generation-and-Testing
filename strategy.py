#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 12h EMA50 trend filter and volume confirmation.
Long when Alligator is bullish (Lips > Teeth > Jaw) AND price > 12h EMA50 AND volume > 1.5x 20-period MA.
Short when Alligator is bearish (Lips < Teeth < Jaw) AND price < 12h EMA50 AND volume > 1.5x 20-period MA.
Exit when Alligator direction reverses or price crosses 12h EMA50.
Uses 12h HTF for trend filter to avoid counter-trend trades, Alligator for trend identification, volume for confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Alligator identifies trends early with smoothed MAs, 12h EMA50 filters major trend, volume avoids low-momentum signals.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
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
    
    # Calculate Williams Alligator (6h timeframe)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Smoothed moving averages (SMMA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator lines
    median_price = (high + low) / 2
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Apply shifts (Alligator specific)
    jaw_shifted = np.roll(jaw, jaw_shift)
    teeth_shifted = np.roll(teeth, teeth_shift)
    lips_shifted = np.roll(lips, lips_shift)
    
    # Set NaN for shifted values that rolled from beginning
    jaw_shifted[:jaw_shift] = np.nan
    teeth_shifted[:teeth_shift] = np.nan
    lips_shifted[:lips_shift] = np.nan
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need enough for Alligator calculation + shifts + EMA + volume MA
    start_idx = max(
        lips_period + lips_shift - 1,  # Alligator ready
        50,  # EMA50
        20   # Volume MA
    )
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Alligator conditions
        alligator_bullish = lips_val > teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val < jaw_val
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Alligator bullish AND price > 12h EMA50 AND volume filter
            if alligator_bullish and price > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND price < 12h EMA50 AND volume filter
            elif alligator_bearish and price < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator turns bearish OR price < 12h EMA50
                if not alligator_bullish or price < ema_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator turns bullish OR price > 12h EMA50
                if not alligator_bearish or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsAlligator_12hEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0