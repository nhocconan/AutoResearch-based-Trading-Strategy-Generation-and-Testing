#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_fractal_1d_trend"
timeframe = "6h"
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
    
    # Get 1d data for trend filter and Williams Fractal
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Fractal on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n+1] < high[n-1] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n+1] > low[n-1] < low[n+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    # Need at least 5 points for fractal (2 on each side)
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i+1] < high_1d[i-1] and 
            high_1d[i+2] < high_1d[i-1]):
            bearish_fractal[i] = True
            
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i+1] > low_1d[i-1] and 
            low_1d[i+2] > low_1d[i-1]):
            bullish_fractal[i] = True
    
    # Williams fractal needs 2 extra 1d bars for confirmation (as per rules)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(50, 10) + 2  # EMA warmup + fractal delay
    
    for i in range(start_idx, n):
        # Skip if EMA not available
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bearish fractal appears or price closes below 1d EMA
            if bearish_fractal_aligned[i] > 0.5 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: bullish fractal appears or price closes above 1d EMA
            if bullish_fractal_aligned[i] > 0.5 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: bullish fractal confirmed and price above 1d EMA (uptrend continuation)
            if bullish_fractal_aligned[i] > 0.5 and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish fractal confirmed and price below 1d EMA (downtrend continuation)
            elif bearish_fractal_aligned[i] > 0.5 and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals