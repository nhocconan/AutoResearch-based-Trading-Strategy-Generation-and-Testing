# 6h_1d_ema_cross_volume_filter
# Hypothesis: 6-hour EMA crossover with 1-day EMA filter and volume confirmation
# Uses 6h EMA cross (12/26) for entry timing, filtered by 1-day EMA trend (50)
# Volume > 1.5x average confirms momentum. Works in bull/bear by requiring trend alignment.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "6h_1d_ema_cross_volume_filter"
timeframe = "6h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 6h EMA crossover system (12/26)
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # 1-day EMA trend filter (50-period)
    ema_trend = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_trend_aligned = align_htf_to_ltf(prices, df_1d, ema_trend)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_trend_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i])):
            signals[i] = 0.0
            continue
        
        # Bullish EMA cross: fast crosses above slow
        bullish_cross = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        # Bearish EMA cross: fast crosses below slow
        bearish_cross = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # Long entry: bullish cross + above daily EMA trend + volume
        if (bullish_cross and close[i] > ema_trend_aligned[i] and vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: bearish cross + below daily EMA trend + volume
        elif (bearish_cross and close[i] < ema_trend_aligned[i] and vol_confirm[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal
        elif position == 1 and bearish_cross:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bullish_cross:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals