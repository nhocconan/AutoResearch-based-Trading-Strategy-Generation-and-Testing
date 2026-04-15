#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions. In trending markets (1d EMA50),
# we take counter-trend entries at extreme %R levels with volume confirmation.
# Works in both bull/bear by following 1d trend direction. Target: 20-50 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Uptrend (price > 1d EMA50), oversold (%R < -80), volume spike
        if (close[i] > ema_1d_aligned[i] and 
            williams_r[i] < -80 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Downtrend (price < 1d EMA50), overbought (%R > -20), volume spike
        elif (close[i] < ema_1d_aligned[i] and 
              williams_r[i] > -20 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Trend fails or %R returns to neutral zone
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= ema_1d_aligned[i] or williams_r[i] > -50)) or
               (signals[i-1] == -0.25 and (close[i] >= ema_1d_aligned[i] or williams_r[i] < -50)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WilliamsR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0