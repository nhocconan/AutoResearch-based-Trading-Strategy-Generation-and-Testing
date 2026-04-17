#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d EMA trend filter and volume confirmation.
# Williams Fractals identify potential reversal points with built-in confirmation delay.
# Combines price action signals with trend and volume filters for robustness.
# Designed to work in both bull (catching breakouts from bullish fractals) and bear (short breakdowns from bearish fractals) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for trend filter and fractals ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Fractals on 1d data (requires 2-bar confirmation after center)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    n1d = len(high_1d)
    bearish_fractal = np.zeros(n1d, dtype=bool)
    bullish_fractal = np.zeros(n1d, dtype=bool)
    
    for i in range(2, n1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i-1] < high_1d[i] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i-1] < low_1d[i] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Add 2-bar confirmation delay for fractals (they need 2 future bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # === 12h data for volume confirmation ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # 12h volume average (20-period)
    vol_avg20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg20_12h)
    
    signals = np.zeros(n)
    warmup = 36  # 34 for EMA + 2 for fractal confirmation
    position = 0
    
    for i in range(warmup, n):
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(vol_avg20_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        vol_filter = vol_12h_current > 1.5 * vol_avg20_12h_aligned[i]
        
        if position == 0:
            # Long: bullish fractal breakout + 1d uptrend + volume filter
            if bullish_fractal_aligned[i] > 0.5 and close[i] > ema34_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish fractal breakdown + 1d downtrend + volume filter
            if bearish_fractal_aligned[i] > 0.5 and close[i] < ema34_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long on bearish fractal or trend reversal
            if bearish_fractal_aligned[i] > 0.5 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on bullish fractal or trend reversal
            if bullish_fractal_aligned[i] > 0.5 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0