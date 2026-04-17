# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Fractal breakout with weekly trend filter and volume confirmation.
# Williams Fractals identify local swing highs/lows with confirmation delay. 
# Breakouts from fractal levels capture momentum after consolidation.
# Weekly trend filter ensures trades align with higher-timeframe momentum.
# Volume confirmation validates breakout strength.
# Works in bull/bear markets by capturing breakouts in trending regimes.
# Target: 12-37 trades/year (50-150 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    def calculate_ema(arr, period):
        ema = np.full_like(arr, np.nan)
        if len(arr) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema[i] = (arr[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema34_1w = calculate_ema(close_1w, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Daily data for Williams Fractals ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: bearish (swing high) and bullish (swing low)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    def calculate_williams_fractals(high, low):
        n = len(high)
        bearish = np.full(n, np.nan)
        bullish = np.full(n, np.nan)
        
        for i in range(2, n-2):
            # Bearish fractal (swing high)
            if (high[i-2] < high[i-1] and 
                high[i] > high[i-1] and 
                high[i] > high[i+1] and 
                high[i] > high[i+2]):
                bearish[i] = high[i]
            
            # Bullish fractal (swing low)
            if (low[i-2] > low[i-1] and 
                low[i] < low[i-1] and 
                low[i] < low[i+1] and 
                low[i] < low[i+2]):
                bullish[i] = low[i]
        
        return bearish, bullish
    
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    # Williams fractals need 2-bar confirmation delay after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # === Daily volume for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg20_1d = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_avg20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    signals = np.zeros(n)
    position = 0
    warmup = 100  # Sufficient for all indicators
    
    for i in range(warmup, n):
        if (np.isnan(ema34_1w_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance) + weekly uptrend + volume
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal (support) + weekly downtrend + volume
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below bullish fractal (support) or weekly trend turns down
            if (close[i] < bullish_fractal_aligned[i] or 
                close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above bearish fractal (resistance) or weekly trend turns up
            if (close[i] > bearish_fractal_aligned[i] or 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_WeeklyEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0