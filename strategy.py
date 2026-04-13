#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with volume confirmation and momentum filter.
# The Alligator (Jaw/Teeth/Lips) identifies trends when lines are aligned and separated.
# In trending markets (JAW < TEETH < LIPS for uptrend, reverse for downtrend),
# we enter with volume confirmation and exit when lines re-intertwine.
# Uses 12h primary timeframe with 1-day trend filter for higher reliability.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (Jaw +8, Teeth +5, Lips +3)
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # Calculate average true range for volatility filter
    def atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed ATR
        atr_result = np.zeros_like(tr)
        if len(tr) >= period:
            atr_result[period-1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr_result[i] = (atr_result[i-1] * (period-1) + tr[i]) / period
        return atr_result
    
    atr_values = atr(high, low, close, 14)
    
    # Volume moving average for confirmation
    def sma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        for i in range(period-1, len(arr)):
            result[i] = np.mean(arr[i-period+1:i+1])
        return result
    
    volume_ma = sma(volume, 20)
    
    # Align all indicators to main timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_values)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        atr_val = atr_aligned[i]
        vol_val = volume[i]
        vol_ma_val = volume_ma_aligned[i]
        
        # Alligator alignment conditions
        bullish_alignment = (jaw_val < teeth_val) and (teeth_val < lips_val)
        bearish_alignment = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        # Volume confirmation (volume above average)
        volume_confirmed = vol_val > vol_ma_val
        
        # Volatility filter (avoid extremely low volatility periods)
        vol_filter = atr_val > 0
        
        if position == 0:
            # Long: bullish alignment + volume confirmation
            if bullish_alignment and volume_confirmed and vol_filter:
                position = 1
                signals[i] = position_size
            # Short: bearish alignment + volume confirmation
            elif bearish_alignment and volume_confirmed and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: alignment breaks or volume dries up
            if not bullish_alignment or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: alignment breaks or volume dries up
            if not bearish_alignment or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Alligator_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0