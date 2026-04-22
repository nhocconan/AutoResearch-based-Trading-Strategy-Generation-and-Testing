#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Trend Filter
# Long when Green line > Red line (bullish alignment) + price > Blue line + volume spike
# Short when Red line > Green line (bearish alignment) + price < Blue line + volume spike
# Exit when lines cross in opposite direction or price crosses Blue line
# Williams Alligator uses smoothed moving averages (SMMA) with specific periods
# Designed for low-frequency trading (~15-25/year) with strong trend-following edge
# Works in both bull and bear markets by following the trend direction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Williams Alligator calculation
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Williams Alligator lines (Smoothed Moving Averages)
    # Jaw (Blue): 13-period SMMA, 8 bars ahead
    # Teeth (Red): 8-period SMMA, 5 bars ahead
    # Lips (Green): 5-period SMMA, 3 bars ahead
    
    # Typical price for Alligator calculation
    typical_price = (high_daily + low_daily + close_daily) / 3
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(data, period):
        sma = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smma(typical_price, 13)  # Blue line
    teeth = smma(typical_price, 8)  # Red line
    lips = smma(typical_price, 5)   # Green line
    
    # Apply the forward shift as per Williams Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align Alligator lines to 12h timeframe (previous day's values)
    jaw_aligned = align_htf_to_ltf(prices, df_daily, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_daily, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_daily, lips)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-period average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: Lips > Teeth (bullish alignment) + price > Jaw + volume spike
            if lips_val > teeth_val and price > jaw_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Teeth > Lips (bearish alignment) + price < Jaw + volume spike
            elif teeth_val > lips_val and price < jaw_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: lines cross in opposite direction or price crosses Jaw
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Teeth crosses above Lips (bearish shift) or price < Jaw
                if teeth_val > lips_val or price < jaw_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Lips crosses above Teeth (bullish shift) or price > Jaw
                if lips_val > teeth_val or price > jaw_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0