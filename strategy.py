#!/usr/bin/env python3

"""
Hypothesis: 4-hour Williams Fractal Breakout with 12-hour EMA trend filter and volume confirmation.
Trades breakouts of Williams fractal highs/lows in the direction of the 12-hour EMA trend.
Volume spike confirms institutional participation. Designed for low trade frequency
(15-30 trades/year) to minimize fee drift and work in both bull and bear markets by aligning with
higher timeframe trend and using fractal-based structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams fractal highs and lows.
    Returns (fractal_high, fractal_low) arrays with NaN where no fractal.
    """
    n = len(high)
    fractal_high = np.full(n, np.nan)
    fractal_low = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Fractal high: high[i] is highest of 5-bar window (i-2, i-1, i, i+1, i+2)
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            fractal_high[i] = high[i]
        # Fractal low: low[i] is lowest of 5-bar window
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            fractal_low[i] = low[i]
    
    return fractal_high, fractal_low

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for trend filter and fractal calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # 12-hour EMA for trend filter (21-period)
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 12-hour Williams fractals
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    fractal_high_12h, fractal_low_12h = calculate_williams_fractals(high_12h, low_12h)
    fractal_high_12h_aligned = align_htf_to_ltf(prices, df_12h, fractal_high_12h, additional_delay_bars=2)
    fractal_low_12h_aligned = align_htf_to_ltf(prices, df_12h, fractal_low_12h, additional_delay_bars=2)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(fractal_high_12h_aligned[i]) or 
            np.isnan(fractal_low_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above recent 12h fractal high with uptrend bias
            if close[i] > fractal_high_12h_aligned[i] and close[i] > ema_21_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below recent 12h fractal low with downtrend bias
            elif close[i] < fractal_low_12h_aligned[i] and close[i] < ema_21_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite fractal level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price closes below 12h fractal low or below 12h EMA
                if close[i] < fractal_low_12h_aligned[i] or close[i] < ema_21_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price closes above 12h fractal high or above 12h EMA
                if close[i] > fractal_high_12h_aligned[i] or close[i] > ema_21_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_12hEMA21_Volume"
timeframe = "4h"
leverage = 1.0