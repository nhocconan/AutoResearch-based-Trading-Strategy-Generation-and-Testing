#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams Fractal Breakout with 12-hour EMA trend filter and volume confirmation.
Trades breakouts of daily Williams fractals in the direction of the 12-hour EMA trend.
Uses volume spike to confirm breakout strength. Designed for low trade frequency (10-30 trades/year)
to minimize fee drag and capture momentum in both bull and bear markets by combining
fractal breakouts (structure) with trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_williams_fractals(high, low):
    """Calculate Williams fractals: returns (bearish_fractal, bullish_fractal) arrays."""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest of 5 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        # Bullish fractal: low[i] is lowest of 5 bars
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA for trend filter (34-period)
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_ltf_to_htf(prices, df_12h, ema_34_12h)
    
    # Load 1d data for Williams fractals - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Williams fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    
    # Align fractals with 2-bar delay for confirmation (fractals need 2 bars to confirm)
    bearish_fractal_aligned = align_ltf_to_htf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_ltf_to_htf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above bullish fractal with uptrend bias
            if close[i] > bullish_fractal_aligned[i] and close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal with downtrend bias
            elif close[i] < bearish_fractal_aligned[i] and close[i] < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite fractal or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below bearish fractal or closes below 12h EMA
                if close[i] < bearish_fractal_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above bullish fractal or closes above 12h EMA
                if close[i] > bullish_fractal_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0