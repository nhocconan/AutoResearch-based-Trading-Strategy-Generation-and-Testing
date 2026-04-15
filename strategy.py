#!/usr/bin/env python3
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
    
    # Daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h Donchian Channel (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: current > 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    # ATR-based volatility filter: require ATR > 0.3 * median ATR (more sensitive for 6h)
    atr_median = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=1).median()
    vol_filter = atr_1d_aligned > 0.3 * atr_median
    
    # Weekly pivot direction from daily data
    # Calculate weekly high/low/close from daily data (simplified: use last 5 days)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(vol_filter[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            continue
        
        # Determine trend based on weekly pivot
        bullish_trend = close[i] > weekly_pivot_aligned[i]
        bearish_trend = close[i] < weekly_pivot_aligned[i]
        
        # Long: price breaks above Donchian high + volume + volatility filter + bullish weekly pivot
        if (close[i] > donch_high[i] and volume[i] > vol_threshold[i] and 
            vol_filter[i] and bullish_trend):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + volume + volatility filter + bearish weekly pivot
        elif (close[i] < donch_low[i] and volume[i] > vol_threshold[i] and 
              vol_filter[i] and bearish_trend):
            signals[i] = -0.25
        
        # Exit: price returns to Donchian middle (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_mid[i]) or
               (signals[i-1] == -0.25 and close[i] > donch_mid[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0