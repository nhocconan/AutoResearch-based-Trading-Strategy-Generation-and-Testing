#!/usr/bin/env python3
name = "1d_Keltner_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # Load weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    close_w = df_w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_w_aligned = align_htf_to_ltf(prices, df_w, ema_50_w)
    
    # Daily ATR for Keltner channels
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Daily EMA20 for Keltner center
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels (2x ATR)
    upper = ema_20 + 2 * atr
    lower = ema_20 - 2 * atr
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_w_aligned[i]) or 
            np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner + above weekly EMA50 + volume spike
            if (close[i] > upper[i] and 
                close[i] > ema_50_w_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner + below weekly EMA50 + volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema_50_w_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below EMA20 or below weekly EMA50
            if close[i] < ema_20[i] or close[i] < ema_50_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above EMA20 or above weekly EMA50
            if close[i] > ema_20[i] or close[i] > ema_50_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals