#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with volume confirmation and daily ATR filter
# Weekly Donchian(10) breakout captures strong momentum across multiple market regimes.
# Volume > 1.3x 20-bar median ensures institutional participation.
# Daily ATR(14) filter avoids extreme volatility (exhaustion) and low volatility (chop).
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian(10) channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    high_10 = pd.Series(high_1w).rolling(window=10, min_periods=10).max()
    low_10 = pd.Series(low_1w).rolling(window=10, min_periods=10).min()
    high_10_aligned = align_htf_to_ltf(prices, df_1w, high_10)
    low_10_aligned = align_htf_to_ltf(prices, df_1w, low_10)
    
    # Daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # first TR is NaN
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: current > 1.3x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.3 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after warmup for weekly Donchian
        # Skip if any required data is NaN
        if (np.isnan(high_10_aligned[i]) or np.isnan(low_10_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Avoid extreme volatility (exhaustion) and low volatility (chop)
        atr_median = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).median()
        vol_filter = (atr_14_aligned[i] > 0.3 * atr_median[i]) and (atr_14_aligned[i] < 3.0 * atr_median[i])
        
        # Long: Weekly Donchian breakout up + volume spike + volatility filter
        if (close[i] > high_10_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Weekly Donchian breakout down + volume spike + volatility filter
        elif (close[i] < low_10_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price re-enters weekly Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_10_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > low_10_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "WeeklyDonchianBreakout_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0