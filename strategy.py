#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(55) breakout with volume confirmation and 1d ATR filter
# Uses longer Donchian period to reduce trade frequency and capture stronger trends.
# Volume > 2x median ensures strong institutional participation.
# 1d ATR(14) filter avoids low volatility (chop) and exhaustion volatility.
# Designed for trend following in both bull and bear markets with conservative sizing.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(55) channels - longer period for fewer, stronger signals
    high_55 = pd.Series(high).rolling(window=55, min_periods=55).max()
    low_55 = pd.Series(low).rolling(window=55, min_periods=55).min()
    
    # 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: current > 2x median of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(55, n):  # Start after warmup for Donchian(55)
        # Skip if any required data is NaN
        if (np.isnan(high_55[i]) or np.isnan(low_55[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extremes
        atr_median = pd.Series(atr_14_aligned).rolling(window=100, min_periods=100).median()
        vol_filter = (atr_14_aligned[i] > 0.3 * atr_median[i]) and (atr_14_aligned[i] < 3.0 * atr_median[i])
        
        # Long: Donchian breakout up + volume spike + volatility filter
        if (close[i] > high_55[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Donchian breakout down + volume spike + volatility filter
        elif (close[i] < low_55[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price re-enters Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_55[i]) or
               (signals[i-1] == -0.25 and close[i] > low_55[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DonchianBreakout55_Volume2x_ATRFilter"
timeframe = "4h"
leverage = 1.0