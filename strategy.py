# 6h_volume_confirmation_breakout_1d_trend_v1
# Hypothesis: 6h volume-confirmed breakouts of 20-period high/low, filtered by 1d EMA50 trend, capture momentum with low turnover.
# Volume filter ensures institutional participation. Trend filter avoids counter-trend trades. Targets 15-25 trades/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_confirmation_breakout_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period high/low for breakout
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x average volume
        vol_confirm = volume[i] > 1.8 * vol_avg_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 1d EMA50
            if close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above 1d EMA50
            if close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above 20-period high + volume confirmation + uptrend
            if (close[i] > high_max_20[i] and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below 20-period low + volume confirmation + downtrend
            elif (close[i] < low_min_20[i] and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals