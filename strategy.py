#!/usr/bin/env python3
# 1d_Weekly_Trend_Filtered_Breakout_v1
# Hypothesis: On daily timeframe, buy when price breaks above weekly Donchian high
# with weekly EMA trend alignment and volume confirmation; sell when price breaks
# below weekly Donchian low with trend alignment. Weekly filter reduces whipsaw
# in sideways markets while capturing strong trends. Works in bull/bear by
# following higher timeframe trend. Target: 20-50 trades over 4 years (5-12/year).

name = "1d_Weekly_Trend_Filtered_Breakout_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high = high_20
    donchian_low = low_20
    
    # Weekly EMA(50) for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50 = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Start after sufficient lookback
    start_idx = 50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Price breaks above weekly Donchian high, above weekly EMA(50), volume above average
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume[i] > vol_ma[i]):
                position = 1
                signals[i] = 0.25
            # Short: Price breaks below weekly Donchian low, below weekly EMA(50), volume above average
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume[i] > vol_ma[i]):
                position = -1
                signals[i] = -0.25
    
    return signals