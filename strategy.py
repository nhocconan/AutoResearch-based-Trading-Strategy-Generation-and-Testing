#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout + Volume Spike + 1w Trend Filter
# Long when price breaks above Donchian(20) high + volume spike + 1w EMA50 uptrend
# Short when price breaks below Donchian(20) low + volume spike + 1w EMA50 downtrend
# Works in bull (strong breakouts with volume) and bear (breakdowns with volume)
# Uses discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current > 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above Donchian high + volume spike + 1w uptrend
        if (close[i] > high_roll[i-1] and 
            volume[i] > vol_threshold[i] and 
            close[i] > ema_1w_aligned[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below Donchian low + volume spike + 1w downtrend
        elif (close[i] < low_roll[i-1] and 
              volume[i] > vol_threshold[i] and 
              close[i] < ema_1w_aligned[i]):
            signals[i] = -0.25
        
        # Exit: Price retrace to midpoint or trend fails
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= (high_roll[i-1] + low_roll[i-1]) / 2 or close[i] < ema_1w_aligned[i])) or
               (signals[i-1] == -0.25 and (close[i] >= (high_roll[i-1] + low_roll[i-1]) / 2 or close[i] > ema_1w_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0