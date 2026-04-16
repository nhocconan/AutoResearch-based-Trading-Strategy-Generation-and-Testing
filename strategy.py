#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Williams Alligator (Jaw/Teeth/Lips) + 1d ATR-based volatility regime filter.
# Long when Lips > Teeth > Jaw (bullish alignment) AND ATR(14) < ATR(50) (low volatility regime).
# Short when Lips < Teeth < Jaw (bearish alignment) AND ATR(14) < ATR(50).
# Exit when Alligator alignment breaks or ATR(14) > ATR(50) (high volatility regime).
# Uses discrete position size 0.30. Alligator identifies trends without lag; ATR regime filter avoids whipsaws in chop.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (capture uptrends) and bear markets (capture downtrends) by following Alligator alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d Indicators: ATR(14) and ATR(50) for volatility regime ===
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50_1d = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h Indicators: Williams Alligator (SMA-based) ===
    # Jaw = SMA(13, 8) -> median price smoothed, shifted 8 bars
    # Teeth = SMA(8, 5) -> median price smoothed, shifted 5 bars
    # Lips = SMA(5, 3) -> median price smoothed, shifted 3 bars
    median_price = (high + low) / 2
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Shift to align with Alligator definition (future-looking smoothing)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Invalidate shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align all indicators to primary timeframe (4h)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)  # df_1d used as proxy for alignment structure
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # ATR50 and Alligator shifts need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        atr14 = atr14_aligned[i]
        atr50 = atr50_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Alligator alignment breaks (not Lips>Teeth>Jaw) OR high volatility (ATR14>ATR50)
            if not (lips_val > teeth_val and teeth_val > jaw_val) or (atr14 > atr50):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Alligator alignment breaks (not Lips<Teeth<Jaw) OR high volatility (ATR14>ATR50)
            if not (lips_val < teeth_val and teeth_val < jaw_val) or (atr14 > atr50):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) AND low volatility (ATR14 < ATR50)
            if (lips_val > teeth_val and teeth_val > jaw_val) and (atr14 < atr50):
                signals[i] = 0.30
                position = 1
            
            # SHORT: Lips < Teeth < Jaw (bearish alignment) AND low volatility (ATR14 < ATR50)
            elif (lips_val < teeth_val and teeth_val < jaw_val) and (atr14 < atr50):
                signals[i] = -0.30
                position = -1
        
        else:
            signals[i] = position * 0.30  # maintain position
    
    return signals

name = "4h_WilliamsAlligator_1dATRRegimeFilter_V1"
timeframe = "4h"
leverage = 1.0