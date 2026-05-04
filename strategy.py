#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend identification, 1d EMA50 for higher timeframe trend alignment,
# and volume spike for confirmation. Designed for 12-30 trades/year on 12h timeframe to minimize fee drag.
# Works in bull markets via trend continuation signals and in bear markets via trend exhaustion/reversal signals.
# The 1d EMA50 provides a robust trend filter that avoids whipsaw during choppy regimes.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Williams Alligator on 12h timeframe (using Smoothed Moving Average - SMMA)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    # Calculate Smoothed Moving Average (SMMA) - equivalent to RMA/Wilder's MA
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply Alligator shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align Alligator lines to LTF (already accounts for completed bar timing)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)  # Using 1d as reference for alignment timing
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: 25-period EMA of volume on 12h timeframe
    vol_ema_25 = pd.Series(volume).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ema_25[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND price above 1d EMA50 AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > (2.0 * vol_ema_25[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND price below 1d EMA50 AND volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > (2.0 * vol_ema_25[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips < Teeth) OR price closes below 1d EMA50
            if lips_aligned[i] < teeth_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips > Teeth) OR price closes above 1d EMA50
            if lips_aligned[i] > teeth_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals