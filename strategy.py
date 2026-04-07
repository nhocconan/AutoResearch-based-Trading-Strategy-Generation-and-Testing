#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h Trend and Volume
Long when price breaks above Donchian(20) high with 12h uptrend and volume confirmation
Short when price breaks below Donchian(20) low with 12h downtrend and volume confirmation
Exit when price crosses back through Donchian midline or trend reverses
Trend following with volatility breakout and volume filter to reduce whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_channel = (highest_high + lowest_low) / 2
    
    # === 12h Trend (HMA 21) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate HMA on 12h data
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        # Align arrays
        wma_half_pad = np.full(len(arr), np.nan)
        wma_full_pad = np.full(len(arr), np.nan)
        wma_half_pad[half-1:half-1+len(wma_half)] = wma_half
        wma_full_pad[period-1:period-1+len(wma_full)] = wma_full
        diff = 2 * wma_half_pad - wma_full_pad
        return wma(diff, sqrt)
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below midline OR 12h trend turns down
            if close[i] < mid_channel[i] or hma_12h_aligned[i] < close_12h[0]:  # simplified trend check
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midline OR 12h trend turns up
            if close[i] > mid_channel[i] or hma_12h_aligned[i] > close_12h[0]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with 12h trend alignment
            if close[i] > highest_high[i] and hma_12h_aligned[i] > close_12h[0]:
                # Break above upper band with uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and hma_12h_aligned[i] < close_12h[0]:
                # Break below lower band with downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals