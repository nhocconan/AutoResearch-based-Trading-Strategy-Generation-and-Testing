#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d Elder Ray and volume confirmation
# Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Strategy: Go long when Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND volume > 1.5x median
# Go short when Lips < Teeth < Jaw (bearish alignment) AND Bear Power > 0 AND volume > 1.5x median
# Exit when alignment breaks or power crosses zero
# Designed to trend-follow in strong moves and avoid whipsaws in chop via Elder Ray filter
# Works in bull (captures trends) and bear (avoids false reversals via power filter)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d close for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6-hour Williams Alligator
    # Smoothed Moving Average (SMMA) - using Wilder's smoothing (alpha = 1/period)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (prev SMMA * (period-1) + current price) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Jaw: 13-period SMMA shifted 8 bars
    jaw_raw = smma(close, 13)
    jaw = np.roll(jaw_raw, 8)  # shift right by 8
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth_raw = smma(close, 8)
    teeth = np.roll(teeth_raw, 5)  # shift right by 5
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips: 5-period SMMA shifted 3 bars
    lips_raw = smma(close, 5)
    lips = np.roll(lips_raw, 3)  # shift right by 3
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Long: bullish alignment + bull power positive + volume confirmation
        if bullish_alignment and bull_power_1d_aligned[i] > 0 and volume[i] > vol_threshold[i]:
            signals[i] = 0.25
        
        # Short: bearish alignment + bear power positive + volume confirmation
        elif bearish_alignment and bear_power_1d_aligned[i] > 0 and volume[i] > vol_threshold[i]:
            signals[i] = -0.25
        
        # Exit: alignment breaks or power crosses zero
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and 
                (not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or bull_power_1d_aligned[i] <= 0)) or
               (signals[i-1] == -0.25 and 
                (not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or bear_power_1d_aligned[i] <= 0)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Alligator_ElderRay_Volume"
timeframe = "6h"
leverage = 1.0