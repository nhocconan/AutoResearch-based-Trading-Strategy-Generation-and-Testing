#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams fractal and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Williams Fractal (bearish: peak, bullish: trough)
    high_arr = df_1d['high'].values
    low_arr = df_1d['low'].values
    bearish_fractal = np.zeros(len(high_arr))
    bullish_fractal = np.zeros(len(low_arr))
    
    for i in range(2, len(high_arr) - 2):
        if (high_arr[i] > high_arr[i-1] and high_arr[i] > high_arr[i-2] and
            high_arr[i] > high_arr[i+1] and high_arr[i] > high_arr[i+2]):
            bearish_fractal[i] = high_arr[i]
        if (low_arr[i] < low_arr[i-1] and low_arr[i] < low_arr[i-2] and
            low_arr[i] < low_arr[i+1] and low_arr[i] < low_arr[i+2]):
            bullish_fractal[i] = low_arr[i]
    
    # Fractals need 2 extra daily bars for confirmation (per rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike detection (6h timeframe)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Require strong volume spike
        
        if position == 0:
            # Long: Bullish fractal breakout with 1d uptrend and volume spike
            if close[i] > bullish_fractal_aligned[i] and close[i] > ema_1d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal breakdown with 1d downtrend and volume spike
            elif close[i] < bearish_fractal_aligned[i] and close[i] < ema_1d_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below bearish fractal or trend turns down
            if close[i] < bearish_fractal_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above bullish fractal or trend turns up
            if close[i] > bullish_fractal_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals