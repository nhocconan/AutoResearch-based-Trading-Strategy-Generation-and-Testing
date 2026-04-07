#!/usr/bin/env python3
"""
1d_breakout_volume_trend_v1
Hypothesis: On daily timeframe, enter long when price breaks above 20-day high with above-average volume and weekly trend bullish, enter short when price breaks below 20-day low with above-average volume and weekly trend bearish. Exit when price crosses 10-day EMA. Designed for 10-25 trades/year to minimize fee decay while capturing strong momentum moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-day high/low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-day EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter (1w EMA of close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_10[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 10-day EMA
            if close[i] < ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 10-day EMA
            if close[i] > ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above 20-day high and weekly trend bullish
                if close[i] > high_20[i] and close[i] > ema_20_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below 20-day low and weekly trend bearish
                elif close[i] < low_20[i] and close[i] < ema_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals