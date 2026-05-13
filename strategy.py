#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + 1d EMA34 trend filter + volume spike confirmation.
# Long when price > Alligator Jaw AND Alligator Teeth > Alligator Lips (bullish alignment) AND close > 1d EMA34 AND volume > 1.8x average
# Short when price < Alligator Jaw AND Alligator Teeth < Alligator Lips (bearish alignment) AND close < 1d EMA34 AND volume > 1.8x average
# Exit when Alligator alignment reverses OR price crosses 1d EMA34
# Uses 12h timeframe for lower trade frequency, Williams Alligator for trend structure, 1d EMA for higher timeframe trend filter, volume spike for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via trend continuation, bear via counter-trend rallies.

name = "12h_WilliamsAlligator_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    if len(close_12h) >= 13:
        # Smoothed Moving Average (SMMA) approximation using EMA with alpha=1/period
        def smma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan, dtype=float)
            result = np.full_like(arr, np.nan, dtype=float)
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        jaw_raw = smma(close_12h, 13)
        teeth_raw = smma(close_12h, 8)
        lips_raw = smma(close_12h, 5)
        
        # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
        jaw_12h = np.roll(jaw_raw, 8)
        teeth_12h = np.roll(teeth_raw, 5)
        lips_12h = np.roll(lips_raw, 3)
         # Set NaN for shifted values that roll in invalid data
        jaw_12h[:8] = np.nan
        teeth_12h[:5] = np.nan
        lips_12h[:3] = np.nan
    else:
        jaw_12h = np.full_like(high_12h, np.nan)
        teeth_12h = np.full_like(low_12h, np.nan)
        lips_12h = np.full_like(high_12h, np.nan)
    
    # Align Alligator lines to 12h timeframe (already aligned since calculated on 12h)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 12h volume > 1.8x 20-period average (spike confirmation)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for Alligator and EMA
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > jaw AND teeth > lips (bullish alignment) AND close > 1d EMA34 AND volume spike
            if (close[i] > jaw_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price < jaw AND teeth < lips (bearish alignment) AND close < 1d EMA34 AND volume spike
            elif (close[i] < jaw_aligned[i] and 
                  teeth_aligned[i] < lips_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment reverses (teeth <= lips) OR trend reversal (close < 1d EMA34)
            if (teeth_aligned[i] <= lips_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment reverses (teeth >= lips) OR trend reversal (close > 1d EMA34)
            if (teeth_aligned[i] >= lips_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals