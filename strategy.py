#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ehlers Fisher Transform + 12h EMA50 Trend + Volume Spike
# Fisher Transform identifies turning points in price cycles, effective in both trending and ranging markets.
# Long when Fisher crosses above -1.5 with 12h EMA50 uptrend and volume spike.
# Short when Fisher crosses below +1.5 with 12h EMA50 downtrend and volume spike.
# Volume confirmation requires > 1.5x 20-bar median volume.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.
# Designed to capture reversals in bear markets and continuations in bull markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Ehlers Fisher Transform on 6h close (period=10)
    def fishert(price, length=10):
        # Normalize price to [-1, 1] range
        highest = pd.Series(price).rolling(window=length, min_periods=1).max()
        lowest = pd.Series(price).rolling(window=length, min_periods=1).min()
        value1 = 0.33 * 2 * ((price - lowest) / (highest - lowest + 1e-10) - 0.5)
        # Smooth value1
        value2 = pd.Series(value1).ewm(alpha=0.5, adjust=False).mean()
        # Fisher transform
        fish = np.where(np.abs(value2) < 0.99, 0.5 * np.log((1 + value2) / (1 - value2)), 0.5 * np.log((1 + 0.99) / (1 - 0.99)))
        return fish
    
    fish = fishert(close, 10)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(fish[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Fisher crosses above -1.5, 12h EMA50 uptrend, volume spike
        if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
            close[i] > ema_50_12h_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Fisher crosses below +1.5, 12h EMA50 downtrend, volume spike
        elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
              close[i] < ema_50_12h_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Fisher crosses zero or volume drops
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (fish[i] < 0 or volume[i] <= vol_threshold[i])) or
               (signals[i-1] == -0.25 and (fish[i] > 0 or volume[i] <= vol_threshold[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Fisher_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0