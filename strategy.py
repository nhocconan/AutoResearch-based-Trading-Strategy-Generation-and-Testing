#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with daily EMA trend filter and volume confirmation
# Uses daily Williams Fractals to identify key support/resistance levels
# Enters on breakout of fractal levels with trend filter (price vs daily EMA50)
# Volume confirmation reduces false breakouts
# Works in both bull and bear markets via trend filter
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Williams Fractals and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (5-bar window: 2 bars each side)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] > high[n+1] < high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] < low[n+1] > low[n+2]
    n1d = len(high_1d)
    bearish_fractal = np.full(n1d, np.nan)
    bullish_fractal = np.full(n1d, np.nan)
    
    for i in range(2, n1d - 2):
        # Bearish fractal (peak)
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i+1] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal (trough)
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i+1] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Calculate 50-period EMA on daily close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Align fractals and EMA to 6-hour timeframe
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above bearish fractal (resistance) + volume spike + uptrend
            if (close[i] > bearish_fractal_aligned[i] and vol_spike[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bullish fractal (support) + volume spike + downtrend
            elif (close[i] < bullish_fractal_aligned[i] and vol_spike[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite fractal level
            if position == 1:
                if close[i] < bullish_fractal_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > bearish_fractal_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0