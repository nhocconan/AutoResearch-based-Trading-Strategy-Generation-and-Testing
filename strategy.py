#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# Long when price breaks above 12h Donchian high AND price > 1w EMA50 (bullish trend)
# Short when price breaks below 12h Donchian low AND price < 1w EMA50 (bearish trend)
# Volume confirmation: current volume > 1.5x 20-bar median volume
# Exit: price crosses back through Donchian midpoint
# Designed to capture trending moves with volume confirmation while avoiding whipsaws.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(donch_mid_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian high, price > 1w EMA50, volume spike
        if (close[i] > donch_high_aligned[i] and 
            close[i] > ema_1w_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low, price < 1w EMA50, volume spike
        elif (close[i] < donch_low_aligned[i] and 
              close[i] < ema_1w_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back through Donchian midpoint
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] <= donch_mid_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] >= donch_mid_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0