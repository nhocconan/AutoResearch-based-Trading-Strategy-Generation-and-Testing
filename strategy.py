#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d EMA trend filter and volume confirmation.
# Williams Fractals identify potential reversal points - bearish fractal (sell signal) when 
# high is higher than two bars on each side, bullish fractal (buy signal) when low is lower 
# than two bars on each side. Breakouts from these levels with volume confirmation and 
# trend alignment (1d EMA50) capture momentum. Works in both bull/bear markets by using 
# EMA filter to avoid counter-trend trades and requiring volume to confirm breakout strength.
# Target: 20-40 trades/year by requiring fractal formation, volume confirmation, and trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n] > high[n-2], high[n] > high[n-1], high[n] > high[n+1], high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2], low[n] < low[n-1], low[n] < low[n+1], low[n] < low[n+2]
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Align fractals to 12h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Calculate 20-period Donchian channels on 12h data for breakout levels
    high_roll = prices['high'].rolling(window=20, min_periods=20).max()
    low_roll = prices['low'].rolling(window=20, min_periods=20).min()
    upper = high_roll.values
    lower = low_roll.values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend filter: price above/below EMA50
        price_above_ema = price > ema_50_aligned[i]
        price_below_ema = price < ema_50_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: bullish fractal breakout above upper Donchian in uptrend
                if bullish_fractal_aligned[i] > 0.5 and price > upper[i] and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish fractal breakdown below lower Donchian in downtrend
                elif bearish_fractal_aligned[i] > 0.5 and price < lower[i] and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below lower Donchian or trend turns against position
                if price < lower[i] or not price_above_ema:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above upper Donchian or trend turns against position
                if price > upper[i] or not price_below_ema:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0