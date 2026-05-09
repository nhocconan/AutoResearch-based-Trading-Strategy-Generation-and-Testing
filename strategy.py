#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_WilliamsFractal_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Williams Fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals (requires 5-point pattern)
    high_arr = df_1d['high'].values
    low_arr = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_arr), dtype=bool)
    bullish_fractal = np.zeros(len(low_arr), dtype=bool)
    
    for i in range(2, len(high_arr) - 2):
        # Bearish fractal: high[i] is highest of 5 candles
        if (high_arr[i] > high_arr[i-1] and high_arr[i] > high_arr[i-2] and
            high_arr[i] > high_arr[i+1] and high_arr[i] > high_arr[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: low[i] is lowest of 5 candles
        if (low_arr[i] < low_arr[i-1] and low_arr[i] < low_arr[i-2] and
            low_arr[i] < low_arr[i+1] and low_arr[i] < low_arr[i+2]):
            bullish_fractal[i] = True
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 1d volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align to 4h with extra delay for fractals (need 2 bars confirmation)
    bearish_fractal_4h = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_4h = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_4h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_4h[i]) or np.isnan(bullish_fractal_4h[i]) or
            np.isnan(ema50_1d_4h[i]) or np.isnan(volume_filter_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bearish = bearish_fractal_4h[i] > 0.5
        bullish = bullish_fractal_4h[i] > 0.5
        trend = ema50_1d_4h[i]
        vol_filter = volume_filter_4h[i]
        
        if position == 0:
            # Enter long: bullish fractal with volume and above trend
            if bullish and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish fractal with volume and below trend
            elif bearish and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA50
            if close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA50
            if close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals