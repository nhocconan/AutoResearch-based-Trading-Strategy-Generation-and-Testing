#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals (5-bar)
    high_arr = df_1d['high'].values
    low_arr = df_1d['low'].values
    n1d = len(high_arr)
    bearish_fractal = np.zeros(n1d, dtype=bool)
    bullish_fractal = np.zeros(n1d, dtype=bool)
    
    for i in range(2, n1d - 2):
        if (high_arr[i] > high_arr[i-1] and high_arr[i] > high_arr[i-2] and
            high_arr[i] > high_arr[i+1] and high_arr[i] > high_arr[i+2]):
            bearish_fractal[i] = True
        if (low_arr[i] < low_arr[i-1] and low_arr[i] < low_arr[i-2] and
            low_arr[i] < low_arr[i+1] and low_arr[i] < low_arr[i+2]):
            bullish_fractal[i] = True
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current 1d volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align all to 6h with extra delay for fractals (need 2 bars for confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        trend = ema34_1d_aligned[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: bullish fractal break with volume and above trend
            if bull_fract and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish fractal break with volume and below trend
            elif bear_fract and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish fractal appears (trend weakness)
            if bear_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish fractal appears (trend weakness)
            if bull_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals