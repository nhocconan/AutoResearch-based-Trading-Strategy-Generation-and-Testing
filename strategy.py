#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    """
    6h Williams Fractal Breakout with 1d trend filter.
    - Long: Bullish fractal break above resistance with 1d uptrend
    - Short: Bearish fractal break below support with 1d downtrend
    - Exit: Opposite fractal break or trend reversal
    - Uses 2-bar confirmation for fractals to avoid look-ahead
    - Target: 15-35 trades/year on 6h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams fractals on 1d
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(df_1d), dtype=bool)
    bullish_fractal = np.zeros(len(df_1d), dtype=bool)
    
    for i in range(2, len(df_1d) - 2):
        # Bearish fractal (peak)
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = True
        
        # Bullish fractal (trough)
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = True
    
    # Align fractals with 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish fractal break and 1d uptrend (price > EMA50)
            if bullish_fractal_aligned[i] > 0.5 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal break and 1d downtrend (price < EMA50)
            elif bearish_fractal_aligned[i] > 0.5 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish fractal break or trend reversal (price < EMA50)
            if bearish_fractal_aligned[i] > 0.5 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish fractal break or trend reversal (price > EMA50)
            if bullish_fractal_aligned[i] > 0.5 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals