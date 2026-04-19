#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation
# Williams Fractals identify key support/resistance levels with lag confirmation
# 1d EMA50 provides higher timeframe bias to avoid counter-trend trades
# Volume confirmation filters weak breakouts and confirms strength
# Target: 75-200 total trades over 4 years (19-50/year) with disciplined entries
name = "4h_WilliamsFractal_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Fractal detection
    def williams_fractals(high_arr, low_arr):
        n = len(high_arr)
        bullish = np.zeros(n, dtype=bool)
        bearish = np.zeros(n, dtype=bool)
        for i in range(2, n-2):
            if (high_arr[i] > high_arr[i-1] and high_arr[i] > high_arr[i-2] and
                high_arr[i] > high_arr[i+1] and high_arr[i] > high_arr[i+2]):
                bearish[i] = True
            if (low_arr[i] < low_arr[i-1] and low_arr[i] < low_arr[i-2] and
                low_arr[i] < low_arr[i+1] and low_arr[i] < low_arr[i+2]):
                bullish[i] = True
        return bearish, bullish
    
    bearish_fractal, bullish_fractal = williams_fractals(high, low)
    # Need 2-bar confirmation for fractals (per rule 2b)
    bearish_fractal_confirmed = np.zeros(n, dtype=bool)
    bullish_fractal_confirmed = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if bearish_fractal[i-2]:
            bearish_fractal_confirmed[i] = True
        if bullish_fractal[i-2]:
            bullish_fractal_confirmed[i] = True
    
    bearish_fractal_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), 
                                               bearish_fractal_confirmed.astype(float), additional_delay_bars=0)
    bullish_fractal_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), 
                                               bullish_fractal_confirmed.astype(float), additional_delay_bars=0)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish fractal breakout + above 1d EMA50 + volume confirmation
            if (bullish_fractal_aligned[i] > 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal breakdown + below 1d EMA50 + volume confirmation
            elif (bearish_fractal_aligned[i] > 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if bearish fractal forms or breaks below 1d EMA50
            if (bearish_fractal_aligned[i] > 0) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if bullish fractal forms or breaks above 1d EMA50
            if (bullish_fractal_aligned[i] > 0) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals