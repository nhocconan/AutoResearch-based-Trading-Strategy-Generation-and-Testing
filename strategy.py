#!/usr/bin/env python3
"""
6h_12h_Aroon_Breakout_Volume
Hypothesis: Aroon oscillator identifies strong trends. Use Aroon(25) > 50 for uptrend bias,
Aroon(25) < -50 for downtrend bias on 12h timeframe. Enter on 6h breakouts of 20-bar high/low
in direction of 12h trend with volume > 1.5x 20-period average. Exit when Aroon reverses
(goes from >50 to <50 for long exit, <-50 to >-50 for short exit) or price crosses 10-bar
opposite EMA. Works in bull/bear by following 12h trend. Volume confirmation filters weak
breakouts. Target: 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data once for Aroon
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Aroon oscillator: Aroon Up - Aroon Down
    # Aroon Up = ((n - periods since highest high) / n) * 100
    # Aroon Down = ((n - periods since lowest low) / n) * 100
    period = 25
    aroon_up = np.full_like(high_12h, np.nan)
    aroon_down = np.full_like(low_12h, np.nan)
    
    for i in range(period, len(high_12h)):
        # Periods since highest high
        highest_high_idx = np.argmax(high_12h[i-period+1:i+1]) + (i - period + 1)
        periods_since_high = i - highest_high_idx
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        
        # Periods since lowest low
        lowest_low_idx = np.argmin(low_12h[i-period+1:i+1]) + (i - period + 1)
        periods_since_low = i - lowest_low_idx
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    aroon_osc = aroon_up - aroon_down  # Range -100 to +100
    
    aroon_osc_aligned = align_htf_to_ltf(prices, df_12h, aroon_osc)
    
    # Precompute 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-bar highest high and lowest low for breakout
    highest_high = pd.Series(high).rolling(window=20, min_periods=1).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=1).min().values
    
    # 10-bar EMA for exit
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume filter: 20-period average
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if Aroon not ready
        if np.isnan(aroon_osc_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        volume_ok = vol > 1.5 * vol_ma if vol_ma > 0 else False
        
        if position == 0:
            # Long: Aroon bullish (>50) + break above 20-bar high + volume
            if aroon_osc_aligned[i] > 50 and price > highest_high[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Aroon bearish (<-50) + break below 20-bar low + volume
            elif aroon_osc_aligned[i] < -50 and price < lowest_low[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Aroon turns bearish (<50) OR price crosses below 10-bar EMA
            if aroon_osc_aligned[i] < 50 or price < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Aroon turns bullish (>-50) OR price crosses above 10-bar EMA
            if aroon_osc_aligned[i] > -50 or price > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_Aroon_Breakout_Volume"
timeframe = "6h"
leverage = 1.0