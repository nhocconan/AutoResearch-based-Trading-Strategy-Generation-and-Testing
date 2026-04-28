#!/usr/bin/env python3
"""
12h_WilliamsFractal_1dTrend_VolumeConfirm
Hypothesis: Uses daily Williams fractal breakouts (bullish/bearish) with 1-day EMA trend filter and volume confirmation.
Works in both bull and bear markets by trading breakouts of confirmed fractal levels with trend alignment.
Targets 15-25 trades/year on 12h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams fractals: bearish (up) and bullish (down)"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest among ±2 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: low[i] is lowest among ±2 bars
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
    
    # Get 1d data for Williams fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams fractals on daily data
    bearish_fractal, bullish_fractal = calculate_williams_fractals(
        df_1d['high'].values, df_1d['low'].values
    )
    # Williams fractal needs 2 extra bars for confirmation (pattern complete after 2 bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: >2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA34 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Breakout conditions
        long_breakout = (not np.isnan(bullish_fractal_aligned[i]) and 
                        close[i] > bullish_fractal_aligned[i] and 
                        vol_confirm and uptrend)
        short_breakout = (not np.isnan(bearish_fractal_aligned[i]) and 
                         close[i] < bearish_fractal_aligned[i] and 
                         vol_confirm and downtrend)
        
        # Exit conditions: opposite fractal level
        long_exit = (not np.isnan(bearish_fractal_aligned[i]) and 
                    close[i] < bearish_fractal_aligned[i])
        short_exit = (not np.isnan(bullish_fractal_aligned[i]) and 
                     close[i] > bullish_fractal_aligned[i])
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsFractal_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0