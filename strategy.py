#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for trend filter and 1d Williams Fractals (bearish/bullish) for breakout entries
# Entry: Long when price breaks above latest bullish fractal AND price > 1d EMA50 (uptrend) AND volume spike
#        Short when price breaks below latest bearish fractal AND price < 1d EMA50 (downtrend) AND volume spike
# Exit: Price crosses 1d EMA50 (trend reversal)
# Williams Fractals require 2-bar confirmation delay after formation
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag
# Works in both bull and bear markets by trading breakouts with 1d trend filter

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
    
    # Calculate 1d Williams Fractals (requires 2-bar confirmation delay)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bullish fractal: low[n] < low[n-1] and low[n] < low[n+1] and low[n+1] < low[n+2] and low[n-1] < low[n-2]
    # Bearish fractal: high[n] > high[n-1] and high[n] > high[n+1] and high[n+1] > high[n+2] and high[n-1] > high[n-2]
    bullish_fractal = np.full(len(high_1d), np.nan)
    bearish_fractal = np.full(len(high_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] and 
            low_1d[i+1] < low_1d[i+2] and low_1d[i-1] < low_1d[i-2]):
            bullish_fractal[i] = low_1d[i]
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] and 
            high_1d[i+1] > high_1d[i+2] and high_1d[i-1] > high_1d[i-2]):
            bearish_fractal[i] = high_1d[i]
    
    # Align fractals to 4h timeframe with 2-bar confirmation delay
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above latest bullish fractal AND price > 1d EMA50 (uptrend) AND volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below latest bearish fractal AND price < 1d EMA50 (downtrend) AND volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price below 1d EMA50 (trend change)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price above 1d EMA50 (trend change)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals