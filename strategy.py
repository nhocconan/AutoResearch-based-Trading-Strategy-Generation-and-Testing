#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with volume confirmation and trend filter
# Uses Williams Alligator (smoothed medians) to identify trend direction.
# Long when price > Jaw and Teeth > Lips (bullish alignment)
# Short when price < Jaw and Teeth < Lips (bearish alignment)
# Requires volume > 1.5x 20-bar median and ADX > 20 on 1d for trend strength
# Works in bull markets (rides uptrends) and bear markets (rides downtrends)
# Target: 50-150 total trades over 4 years

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price_1d = (high_1d + low_1d) / 2
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(arr, period):
        sma = np.full_like(arr, np.nan, dtype=float)
        sma[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaw_raw = smma(median_price_1d, 13)
    teeth_raw = smma(median_price_1d, 8)
    lips_raw = smma(median_price_1d, 5)
    
    # Shift the lines (Jaw: 8 bars, Teeth: 5 bars, Lips: 3 bars)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set NaN for shifted values that don't exist yet
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align Williams Alligator lines and ADX to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: bullish alignment (price > Jaw and Teeth > Lips) + volume + ADX > 20
        if (close[i] > jaw_aligned[i] and 
            teeth_aligned[i] > lips_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx_aligned[i] > 20 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish alignment (price < Jaw and Teeth < Lips) + volume + ADX > 20
        elif (close[i] < jaw_aligned[i] and 
              teeth_aligned[i] < lips_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx_aligned[i] > 20 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite alignment or ADX < 15 (weak trend)
        elif position == 1 and (close[i] < jaw_aligned[i] or adx_aligned[i] < 15):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > jaw_aligned[i] or adx_aligned[i] < 15):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_Volume_ADX"
timeframe = "12h"
leverage = 1.0