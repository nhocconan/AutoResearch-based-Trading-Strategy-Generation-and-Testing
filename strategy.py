#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsAlligator_Signal_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d: Williams Alligator ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Jaw (blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (red): 8-period SMMA, shifted 5 bars forward  
    # Lips (green): 5-period SMMA, shifted 3 bars forward
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        sma = np.mean(data[:period])
        result[period-1] = sma
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift jaws forward by 8, teeth by 5, lips by 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # For the shifted periods, use original values to avoid look-ahead
    jaw_shifted[:8] = jaw[:8] if len(jaw) > 8 else np.nan
    teeth_shifted[:5] = teeth[:5] if len(teeth) > 5 else np.nan
    lips_shifted[:3] = lips[:3] if len(lips) > 3 else np.nan
    
    # Alligator signals: 
    # Bullish: Lips > Teeth > Jaw (green > red > blue)
    # Bearish: Jaw > Teeth > Lips (blue > red > green)
    bullish = (lips_shifted > teeth_shifted) & (teeth_shifted > jaw_shifted)
    bearish = (jaw_shifted > teeth_shifted) & (teeth_shifted > lips_shifted)
    
    # Align to 6h timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish.astype(float))
    
    # === 6h: Price action and volume confirmation ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        bullish_val = bullish_aligned[i]
        bearish_val = bearish_aligned[i]
        vol_ma = vol_ma20[i]
        
        # Skip if any value is NaN
        if np.isnan(bullish_val) or np.isnan(bearish_val) or np.isnan(vol_ma):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish alligator alignment + volume confirmation
            if bullish_val > 0.5 and volume[i] > vol_ma * 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alligator alignment + volume confirmation
            elif bearish_val > 0.5 and volume[i] > vol_ma * 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment or price closes below teeth
            if bearish_val > 0.5 or close[i] < teeth_shifted[i] if not np.isnan(teeth_shifted[i]) else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment or price closes above teeth
            if bullish_val > 0.5 or close[i] > teeth_shifted[i] if not np.isnan(teeth_shifted[i]) else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals