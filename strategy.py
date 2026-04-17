#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Williams Alligator (Jaw/Teeth/Lips) + volume confirmation + ATR-based trend filter.
Long when price > Alligator Lips, Lips > Teeth, Teeth > Jaw (bullish alignment) with volume > 1.5x 20-period average.
Short when price < Alligator Lips, Lips < Teeth, Teeth < Jaw (bearish alignment) with volume confirmation.
Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw) or volume drops below average.
Uses 12h timeframe for structure (reduces noise) and 4h for entry timing and volume confirmation.
Williams Alligator identifies trend phases and avoids whipsaws in ranging markets. Designed for medium-term trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams Alligator (Smoothed Medians)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_12h = (high_12h + low_12h) / 2
    
    def smma(data, period):
        """Smoothed Moving Average (similar to Wilder's smoothing)"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_12h = smma(median_12h, 13)
    teeth_12h = smma(median_12h, 8)
    lips_12h = smma(median_12h, 5)
    
    # Apply shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw_12h = np.roll(jaw_12h, 8)
    teeth_12h = np.roll(teeth_12h, 5)
    lips_12h = np.roll(lips_12h, 3)
    # Set shifted values to NaN
    jaw_12h[:8] = np.nan
    teeth_12h[:5] = np.nan
    lips_12h[:3] = np.nan
    
    # Calculate 12h ATR for trend filter (strong trend when ATR rising)
    def atr(high, low, close, period=14):
        """Average True Range"""
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First TR is undefined
        # Wilder's smoothing
        atr_vals = np.full_like(tr, np.nan, dtype=float)
        if len(tr) >= period:
            atr_vals[period] = np.nanmean(tr[1:period+1])  # First ATR
            for i in range(period+1, len(tr)):
                atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr_12h = atr(high_12h, low_12h, close_12h, 14)
    atr_ma_12h = pd.Series(atr_12h).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    atr_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Alligator and ATR MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_12h_aligned[i]) or 
            np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or 
            np.isnan(atr_ma_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: ATR rising (strong trend environment)
        trending = atr_ma_12h_aligned[i] > atr_ma_12h_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish = (lips_12h_aligned[i] > teeth_12h_aligned[i] and 
                      teeth_12h_aligned[i] > jaw_12h_aligned[i])
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish = (lips_12h_aligned[i] < teeth_12h_aligned[i] and 
                      teeth_12h_aligned[i] < jaw_12h_aligned[i])
            
            # Long: bullish alignment with volume and trend confirmation
            if bullish and volume_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment with volume and trend confirmation
            elif bearish and volume_confirmed and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips <= Teeth) OR volume drops
            if (lips_12h_aligned[i] <= teeth_12h_aligned[i] or 
                not volume_confirmed):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips >= Teeth) OR volume drops
            if (lips_12h_aligned[i] >= teeth_12h_aligned[i] or 
                not volume_confirmed):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hWilliamsAlligator_Volume_ATRTrend"
timeframe = "4h"
leverage = 1.0