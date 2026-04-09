#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Alligator with volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) from 1d defines trend structure aligned with 12h timeframe
# Volume confirmation (current 12h volume > 1.5x 30-period average) filters false signals
# Long when price > Lips and Lips > Teeth > Jaw (bullish alignment)
# Short when price < Lips and Lips < Teeth < Jaw (bearish alignment)
# Works in bull/bear: Alligator adapts to trending markets, volume confirms validity
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "12h_1d_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (SMMA = smoothed moving average)
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_PRICE) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_1d = smma(close_1d, 13)
    teeth_1d = smma(close_1d, 8)
    lips_1d = smma(close_1d, 5)
    
    # Shift to avoid look-ahead (Alligator uses future data for smoothing)
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Pre-compute volume confirmation (30-period average for 12h)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_30[i]
        
        if position == 1:  # Long position
            # Exit when Alligator closes (Lips < Teeth or Teeth < Jaw) or price < Lips
            if (lips_aligned[i] < teeth_aligned[i] or 
                teeth_aligned[i] < jaw_aligned[i] or
                close[i] < lips_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Alligator closes (Lips > Teeth or Teeth > Jaw) or price > Lips
            if (lips_aligned[i] > teeth_aligned[i] or 
                teeth_aligned[i] > jaw_aligned[i] or
                close[i] > lips_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Alligator alignment with volume confirmation
            # Bullish: Lips > Teeth > Jaw and price > Lips
            # Bearish: Lips < Teeth < Jaw and price < Lips
            if volume_confirmed:
                if (lips_aligned[i] > teeth_aligned[i] and 
                    teeth_aligned[i] > jaw_aligned[i] and
                    close[i] > lips_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                elif (lips_aligned[i] < teeth_aligned[i] and 
                      teeth_aligned[i] < jaw_aligned[i] and
                      close[i] < lips_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals