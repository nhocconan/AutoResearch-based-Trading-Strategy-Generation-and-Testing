#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation.
Long when price breaks above latest bullish fractal AND weekly EMA34 rising AND volume > 2.0x 20-period MA.
Short when price breaks below latest bearish fractal AND weekly EMA34 falling AND volume > 2.0x 20-period MA.
Exit when price touches opposite fractal level or weekly EMA34 reverses.
Williams Fractals provide natural support/resistance levels that work in ranging and trending markets.
Weekly EMA34 filters major trend to avoid counter-trend trades. Volume confirms breakout strength.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Works in both bull and bear markets by following the higher timeframe trend and fading false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams Fractals
    # Bullish fractal: low[i] is lowest among low[i-2], low[i-1], low[i], low[i+1], low[i+2]
    # Bearish fractal: high[i] is highest among high[i-2], high[i-1], high[i], high[i+1], high[i+2]
    bullish_fractal = np.full(n, np.nan)
    bearish_fractal = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Bullish fractal: current low is lowest of 5-bar window
        if (low[i] <= low[i-1] and low[i] <= low[i-2] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = low[i]
        
        # Bearish fractal: current high is highest of 5-bar window
        if (high[i] >= high[i-1] and high[i] >= high[i-2] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = high[i]
    
    # Forward fill fractal levels to use most recent fractal
    bullish_fractal_ffill = pd.Series(bullish_fractal).ffill().values
    bearish_fractal_ffill = pd.Series(bearish_fractal).ffill().values
    
    # Calculate weekly EMA34 for trend filter (HTF) - needs extra delay for confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Weekly EMA34 needs no extra delay as it's trend-following on the weekly close itself
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20)  # Need sufficient data for fractals, weekly EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bullish_fractal_ffill[i]) or np.isnan(bearish_fractal_ffill[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        bullish_level = bullish_fractal_ffill[i]
        bearish_level = bearish_fractal_ffill[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate weekly EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 2.0x 20-period MA (strong breakout confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above bullish fractal (resistance) AND weekly EMA rising AND volume filter
            if price > bullish_level and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below bearish fractal (support) AND weekly EMA falling AND volume filter
            elif price < bearish_level and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches bearish fractal (support) OR weekly EMA starts falling
                if price < bearish_level or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches bullish fractal (resistance) OR weekly EMA starts rising
                if price > bullish_level or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsFractal_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0