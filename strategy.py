#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Williams Fractal breakout with volume confirmation
# Long when price > Alligator Jaw (teeth > lips) AND bullish fractal confirmed AND volume > 1.5x 6h average
# Short when price < Alligator Jaw (teeth < lips) AND bearish fractal confirmed AND volume > 1.5x 6h average
# Williams Alligator (13,8,5 SMAs with future shifts) acts as dynamic trend filter
# Williams Fractals provide reversal points with built-in confirmation (requires 2 bars after)
# Volume conviction filters false breakouts
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Williams Fractals (requires 2-bar confirmation) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal calculation
    n1 = len(high_1d)
    bearish_fractal = np.zeros(n1, dtype=bool)
    bullish_fractal = np.zeros(n1, dtype=bool)
    
    for i in range(2, n1 - 2):
        # Bearish fractal: high[i-2] < high[i-1] < high[i] > high[i+1] > high[i+2]
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i-1] < high_1d[i] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i+1] > high_1d[i+2]):
            bearish_fractal[i] = True
        
        # Bullish fractal: low[i-2] > low[i-1] > low[i] < low[i+1] < low[i+2]
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i-1] > low_1d[i] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i+1] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Convert to float arrays for alignment (1.0 where fractal exists, 0.0 otherwise)
    bearish_fractal_float = bearish_fractal.astype(float)
    bullish_fractal_float = bullish_fractal.astype(float)
    
    # Williams Fractals need 2 extra bars for confirmation (already calculated above)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_float, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_float, additional_delay_bars=2)
    
    # === 6h Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    
    # Smoothed Moving Average (SMMA) approximation using Wilder's smoothing
    def smma(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_6h, 13)  # Blue line
    teeth = smma(close_6h, 8)  # Red line
    lips = smma(close_6h, 5)   # Green line
    
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # === 6h Volume Confirmation ===
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for Alligator (max period 13) + fractal lookback
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        vol_ma_val = vol_ma_6h_aligned[i]
        
        # Williams Alligator conditions
        # Alligator sleeping: all lines intertwined (no clear trend)
        # Alligator awake: jaws, teeth, lips are ordered
        # For uptrend: Lips > Teeth > Jaw (green > red > blue)
        # For downtrend: Jaw > Teeth > Lips (blue > red > green)
        alligator_up = lips_val > teeth_val and teeth_val > jaw_val
        alligator_down = jaw_val > teeth_val and teeth_val > lips_val
        
        # Volume confirmation: current volume > 1.5x 6h average volume
        vol_confirm = volume[i] > vol_ma_val * 1.5
        
        # === ENTRY LOGIC ===
        if position == 0:
            # Long when: Alligator bullish (Lips>Teeth>Jaw) AND bullish fractal AND volume confirmation
            if alligator_up and bull_fract > 0.5 and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: Alligator bearish (Jaw>Teeth>Lips) AND bearish fractal AND volume confirmation
            elif alligator_down and bear_fract > 0.5 and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # === EXIT LOGIC: Reverse when Alligator changes direction ===
        elif position == 1:
            # Exit long when Alligator turns bearish
            if alligator_down:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Alligator turns bullish
            if alligator_up:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_Fractal_Volume1.5x"
timeframe = "6h"
leverage = 1.0