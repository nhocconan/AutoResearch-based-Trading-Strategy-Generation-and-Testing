#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with 1-day trend filter and volume spike
# Alligator lines act as dynamic support/resistance; price crossing all three lines indicates strong trend.
# Works in bull/bear by following the 1-day trend direction. Volume spike confirms momentum.
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_Alligator_JawTeethLips_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 4h: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # Smoothed Median Price (H+L)/2, then SMMA
    median_price = (high + low) / 2
    
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Blue line
    jaw = np.roll(jaw, 8)         # Shifted by 8 bars
    teeth = smma(median_price, 8) # Red line
    teeth = np.roll(teeth, 5)     # Shifted by 5 bars
    lips = smma(median_price, 5)  # Green line
    lips = np.roll(lips, 3)       # Shifted by 3 bars
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50 = ema50_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Price above all three Alligator lines + 1d uptrend + volume spike
            if (close[i] > jaw_val and close[i] > teeth_val and close[i] > lips_val and
                close[i] > ema50 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Price below all three Alligator lines + 1d downtrend + volume spike
            elif (close[i] < jaw_val and close[i] < teeth_val and close[i] < lips_val and
                  close[i] < ema50 and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Teeth (red line) or 1d trend turns down
            if close[i] < teeth_val or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Teeth (red line) or 1d trend turns up
            if close[i] > teeth_val or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals