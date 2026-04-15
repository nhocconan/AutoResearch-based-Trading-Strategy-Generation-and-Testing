#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams %R + Weekly Trend Filter + Volume Spike
# Williams %R identifies overbought/oversold conditions on daily timeframe.
# Weekly trend filter (EMA50 slope) ensures we trade with the higher timeframe trend.
# Volume confirmation requires > 2x 20-bar median volume to filter weak moves.
# Designed to work in bull markets (buy pullbacks in uptrend) and bear markets (sell bounces in downtrend).
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Williams %R(14) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r.values)
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_slope = np.diff(ema_50, prepend=ema_50[0])
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_slope)
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_slope_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Williams %R oversold (< -80), weekly uptrend (slope > 0), volume spike
        if (williams_r_aligned[i] < -80 and 
            ema_50_slope_aligned[i] > 0 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Williams %R overbought (> -20), weekly downtrend (slope < 0), volume spike
        elif (williams_r_aligned[i] > -20 and 
              ema_50_slope_aligned[i] < 0 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Williams %R returns to neutral range (-50 to -30) or volume drops
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (williams_r_aligned[i] > -50 or volume[i] <= vol_threshold[i])) or
               (signals[i-1] == -0.25 and (williams_r_aligned[i] < -30 or volume[i] <= vol_threshold[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WilliamsR_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0