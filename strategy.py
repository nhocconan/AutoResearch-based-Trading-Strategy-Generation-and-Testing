#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for trend filter and 1d Williams fractals for structure
# Entry logic: Long when price breaks above 1d bullish fractal with volume spike and price > 1d EMA50
#              Short when price breaks below 1d bearish fractal with volume spike and price < 1d EMA50
# Exit logic: Exit when price crosses the 1d EMA50 (trend reversal)
# Works in both bull and bear markets by trading with the 1d trend using fractal structure
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "4h_WilliamsFractal_Breakout_1dEMA50_Volume"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Williams fractals (HTF)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: 5-bar pattern
    # Bearish fractal: high[n-2] is highest of [n-4, n-3, n-2, n-1, n]
    # Bullish fractal: low[n-2] is lowest of [n-4, n-3, n-2, n-1, n]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] == np.max(high_1d[i-2:i+3]) and 
            high_1d[i] >= np.max(high_1d[i-2:i+3])):  # Allow equal highs
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] == np.min(low_1d[i-2:i+3]) and 
            low_1d[i] <= np.min(low_1d[i-2:i+3])):  # Allow equal lows
            bullish_fractal[i] = low_1d[i]
    
    # Align Williams fractals to 4h timeframe (use previous completed 1d bar's levels)
    # Add 2-bar delay for fractal confirmation (needs 2 future bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above 1d bullish fractal AND price > 1d EMA50 (uptrend) AND volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below 1d bearish fractal AND price < 1d EMA50 (downtrend) AND volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1d EMA50 (trend change)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1d EMA50 (trend change)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals