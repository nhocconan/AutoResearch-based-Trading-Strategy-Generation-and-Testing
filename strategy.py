#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d VWAP trend filter and volume confirmation
# Williams Alligator (Jaw, Teeth, Lips) identifies trend direction through SMAs.
# VWAP confirms institutional interest. Volume spike validates breakout strength.
# Works in trending markets by following Alligator alignment. VWAP acts as dynamic support/resistance.
# Discrete position sizing (0.25) reduces churn while maintaining exposure.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for VWAP and Alligator components (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_numerator, vwap_denominator, out=np.full_like(vwap_numerator, np.nan), where=vwap_denominator!=0)
    
    # Williams Alligator components (13,8,5 periods shifted)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        sma = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            sma[period-1] = np.nanmean(arr[:period])
            for i in range(period, len(arr)):
                sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply shifts: Jaw(-8), Teeth(-5), Lips(-3)
    jaw_shifted = np.roll(jaw_raw, 8)
    teeth_shifted = np.roll(teeth_raw, 5)
    lips_shifted = np.roll(lips_raw, 3)
    # Set invalid shifted values to NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price > VWAP + volume spike
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and close[i] > vwap_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + price < VWAP + volume spike
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and close[i] < vwap_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator alignment breaks or price crosses VWAP
            if position == 1:
                # Exit long: Bearish alignment OR price below VWAP
                if (lips_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < jaw_aligned[i] or close[i] < vwap_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Bullish alignment OR price above VWAP
                if (lips_aligned[i] > teeth_aligned[i] or teeth_aligned[i] > jaw_aligned[i] or close[i] > vwap_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dVWAP_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0