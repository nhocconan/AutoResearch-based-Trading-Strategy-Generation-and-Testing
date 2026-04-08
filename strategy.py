#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_fractal_breakout_1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and fractal detection
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Fractals (1d)
    n1 = len(high_1d)
    bearish_fractal = np.zeros(n1, dtype=bool)
    bullish_fractal = np.zeros(n1, dtype=bool)
    
    for i in range(2, n1 - 2):
        if (high_1d[i] >= high_1d[i-1] and high_1d[i] >= high_1d[i-2] and
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] <= low_1d[i-1] and low_1d[i] <= low_1d[i-2] and
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Align fractal signals to 1h timeframe
    bearish_fractal_1h = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float))
    bullish_fractal_1h = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float))
    
    # 1d trend: 34-period EMA (responsive trend)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 1h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1h[i]) or np.isnan(bearish_fractal_1h[i]) or 
            np.isnan(bullish_fractal_1h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
            
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bearish fractal or trend fails
            if bearish_fractal_1h[i] == 1.0 or close[i] < ema_34_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: bullish fractal or trend fails
            if bullish_fractal_1h[i] == 1.0 or close[i] > ema_34_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_34_1h[i]
            bearish = close[i] < ema_34_1h[i]
            
            # Long: bullish fractal + bullish trend + volume
            if (bullish_fractal_1h[i] == 1.0 and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short: bearish fractal + bearish trend + volume
            elif (bearish_fractal_1h[i] == 1.0 and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals