#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray combination
# Uses 6h primary timeframe with Williams Alligator (Jaw/Teeth/Lips) for trend identification
# 1d Elder Ray (Bull Power/Bear Power) measures bull/bear strength relative to 13-period EMA
# Long when: Alligator aligned bullish (Lips > Teeth > Jaw) AND 1d Bull Power > 0 AND price > 6h EMA50
# Short when: Alligator aligned bearish (Lips < Teeth < Jaw) AND 1d Bear Power < 0 AND price < 6h EMA50
# Volume confirmation (1.5x 20-period average) ensures strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in both bull and bear markets by requiring confluence of multiple timeframe indicators

name = "6h_WilliamsAlligator_1dElderRay_Trend_Volume_v1"
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
    
    # Get 1d data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA13 and power calculations
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h Williams Alligator (SMMA = Smoothed Moving Average)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(values, period):
        """Smoothed Moving Average"""
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift Alligator lines as per Williams specification
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that rolled from beginning
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    # 6h EMA50 for additional trend filter
    ema50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema50_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish alignment + Bull Power positive + price > EMA50 + volume spike
            if (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and 
                bull_power_1d_aligned[i] > 0 and 
                close[i] > ema50_6h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment + Bear Power negative + price < EMA50 + volume spike
            elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and 
                  bear_power_1d_aligned[i] < 0 and 
                  close[i] < ema50_6h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator loses bullish alignment OR Bull Power turns negative OR price < EMA50
            if not (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]) or \
               bull_power_1d_aligned[i] <= 0 or \
               close[i] < ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator loses bearish alignment OR Bear Power turns positive OR price > EMA50
            if not (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]) or \
               bear_power_1d_aligned[i] >= 0 or \
               close[i] > ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals