#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h trend filter
# Long when price breaks above 20-period high + volume spike + 12h EMA50 uptrend
# Short when price breaks below 20-period low + volume spike + 12h EMA50 downtrend
# Exit when price returns to midline or trend fails
# Uses discrete sizing (0.25) to limit trades to ~30-50/year

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_roll = (high_roll + low_roll) / 2.0
    
    # Volume confirmation: current > 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(mid_roll[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above upper band + volume + 12h uptrend
        if (close[i] > high_roll[i] and volume[i] > vol_threshold[i] and 
            close[i] > ema_12h_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below lower band + volume + 12h downtrend
        elif (close[i] < low_roll[i] and volume[i] > vol_threshold[i] and 
              close[i] < ema_12h_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to midline or trend fails
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= mid_roll[i] or close[i] <= ema_12h_aligned[i])) or
               (signals[i-1] == -0.25 and (close[i] >= mid_roll[i] or close[i] >= ema_12h_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0