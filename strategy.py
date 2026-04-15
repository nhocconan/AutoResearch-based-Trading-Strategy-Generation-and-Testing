#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h EMA trend + volume spike
# Long when price breaks above Donchian upper band (20) AND 12h EMA(50) rising AND volume > 1.5x median
# Short when price breaks below Donchian lower band (20) AND 12h EMA(50) falling AND volume > 1.5x median
# Exit when price crosses back inside Donchian channel or EMA trend reverses
# Uses conservative sizing (0.25) to limit trade frequency and avoid fee drag
# Designed to work in trending markets with volume confirmation and trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_12h_slope = np.diff(ema_12h_aligned, prepend=ema_12h_aligned[0])
    
    # Donchian Channel (20) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_upper = high_roll.values
    donch_lower = low_roll.values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_slope[i]) or np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above Donchian upper, EMA rising, volume spike
        if (close[i] > donch_upper[i] and 
            ema_12h_slope[i] > 0 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below Donchian lower, EMA falling, volume spike
        elif (close[i] < donch_lower[i] and 
              ema_12h_slope[i] < 0 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Price crosses back inside Donchian channel or EMA trend reverses
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < donch_upper[i] or ema_12h_slope[i] <= 0)) or
               (signals[i-1] == -0.25 and (close[i] > donch_lower[i] or ema_12h_slope[i] >= 0)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0