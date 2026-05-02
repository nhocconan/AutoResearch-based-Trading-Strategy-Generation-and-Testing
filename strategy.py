#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d trend filter and volume confirmation
# Uses 1d Williams Fractals for structure (more reliable than pivots in ranging markets),
# 1d EMA(50) for trend direction to avoid counter-trend whipsaws,
# Volume spike (1.5x 30-period average) for participation confirmation.
# Only takes bullish fractal breaks in uptrend and bearish fractal breaks in downtrend.
# Discrete position sizing 0.25 to minimize fee churn and manage drawdown.
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits.
# Williams Fractals provide natural support/resistance that works in both bull and bear markets.

name = "12h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams Fractals, EMA trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d Williams Fractals (bearish = sell signal, bullish = buy signal)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    # Calculate fractals (need 2 bars on each side)
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = high_1d[i-1]  # Center bar high
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = low_1d[i-1]  # Center bar low
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h volume confirmation (1.5x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and volume MA)
    start_idx = 60  # max(50 for EMA, 30 for volume) + buffer for fractals
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above bullish fractal AND uptrend AND volume confirm
            if (close[i] > bullish_fractal_aligned[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal AND downtrend AND volume confirm
            elif (close[i] < bearish_fractal_aligned[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below bullish fractal OR trend reverses to downtrend
            if (close[i] < bullish_fractal_aligned[i] or 
                not uptrend):  # exited if price closes below 1d EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above bearish fractal OR trend reverses to uptrend
            if (close[i] > bearish_fractal_aligned[i] or 
                not downtrend):  # exited if price closes above 1d EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals