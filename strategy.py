#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day Williams Fractal trend filter and 4-hour Donchian breakout (20-period) with volume confirmation.
# Enters only during 08-20 UTC session. Williams Fractal identifies 1-day swing points to filter trades in trending markets.
# Uses tight conditions to limit trades (~20-40/year) and avoid overtrading. Works in both bull and bear via fractal trend filter.
name = "4h_1d_WilliamsFractal_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Williams Fractal trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: bearish (high) and bullish (low) fractals
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    n_1d = len(high_1d)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Fractal trend: 1 if bullish fractal confirmed, -1 if bearish fractal confirmed, 0 otherwise
    # Need 2 additional bars for confirmation (as per Williams Fractal rules)
    fractal_trend = np.zeros(n_1d)
    for i in range(2, n_1d):
        if bullish_fractal[i-2]:  # bullish fractal confirmed 2 bars later
            fractal_trend[i] = 1
        elif bearish_fractal[i-2]:  # bearish fractal confirmed 2 bars later
            fractal_trend[i] = -1
        else:
            fractal_trend[i] = fractal_trend[i-1]  # carry forward previous trend
    
    # Align fractal trend to 4h timeframe with 2-bar additional delay for confirmation
    fractal_trend_aligned = align_htf_to_ltf(prices, df_1d, fractal_trend, additional_delay_bars=2)
    
    # Get 4h data for Donchian breakout
    high_4h = high
    low_4h = low
    # Donchian channels: 20-period high/low
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(fractal_trend_aligned[i]) or np.isnan(high_20_4h[i]) or 
            np.isnan(low_20_4h[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish fractal trend AND price breaks 4h Donchian high with volume
            if (fractal_trend_aligned[i] == 1 and 
                close[i] > high_20_4h[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal trend AND price breaks 4h Donchian low with volume
            elif (fractal_trend_aligned[i] == -1 and 
                  close[i] < low_20_4h[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if bearish fractal trend or price breaks below 4h Donchian low
            if fractal_trend_aligned[i] == -1 or close[i] < low_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if bullish fractal trend or price breaks above 4h Donchian high
            if fractal_trend_aligned[i] == 1 or close[i] > high_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals