#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and ATR trailing stop
# - Uses 12h Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) for trend direction
# - Confirms with 1d volume > 2.5x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop: exits when price retraces 3.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Williams Alligator identifies trends via SMAs with forward shift (no look-ahead)
# - Volume filter reduces false breakouts, ATR stop manages risk in volatile markets
# - Works in both bull and bear markets by following the trend direction

name = "12h_1d_alligator_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    volume_1d = df_1d['volume'].values
    
    # 1d Volume > 2.5x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.5 * avg_volume_20)
    
    # Align 1d indicators to 12h
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 12h Williams Alligator components
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    # SMMA = smoothed moving average (similar to EMA but with different alpha)
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Using high for Jaw (can also use close)
    teeth = smma(high, 8)  # Using high for Teeth
    lips = smma(high, 5)   # Using high for Lips
    
    # Apply forward shift (no look-ahead: shift forward means we use past values)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set shifted values to NaN for invalid periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # 12h ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_12h[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            atr_12h[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 3.0x ATR from high
            if low[i] <= highest_since_entry - (3.0 * atr_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 3.0x ATR from low
            if high[i] >= lowest_since_entry + (3.0 * atr_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Alligator trend detection:
            # Lips > Teeth > Jaw = uptrend (green)
            # Lips < Teeth < Jaw = downtrend (red)
            # Otherwise = no trend (sleeping)
            
            lips_val = lips_shifted[i]
            teeth_val = teeth_shifted[i]
            jaw_val = jaw_shifted[i]
            
            # Check for strong uptrend
            if (lips_val > teeth_val and teeth_val > jaw_val and
                volume_spike_1d_aligned[i]):
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Check for strong downtrend
            elif (lips_val < teeth_val and teeth_val < jaw_val and
                  volume_spike_1d_aligned[i]):
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals