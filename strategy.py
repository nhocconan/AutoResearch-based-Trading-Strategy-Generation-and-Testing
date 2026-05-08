#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with daily trend filter and volume confirmation
# Uses daily Williams Fractals for swing high/low detection on 12h timeframe.
# Requires 1d EMA34 trend alignment and volume spike to avoid false breakouts.
# Designed for low-frequency, high-conviction trades to minimize fee drag.
# Williams Fractals provide strong support/resistance levels that work in both trending and ranging markets.

name = "12h_WilliamsFractal_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (requires 2-bar lookback/forward for confirmation)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    n1d = len(high_1d)
    bearish_fractal = np.zeros(n1d, dtype=bool)
    bullish_fractal = np.zeros(n1d, dtype=bool)
    
    for i in range(2, n1d - 2):
        # Bearish fractal (sell signal)
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i-1] < high_1d[i] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        
        # Bullish fractal (buy signal)
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i-1] > low_1d[i] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Convert to price levels (use the fractal high/low as support/resistance)
    bearish_level = np.where(bearish_fractal, high_1d, np.nan)
    bullish_level = np.where(bullish_fractal, low_1d, np.nan)
    
    # Forward fill to maintain the level until next fractal
    bearish_level_series = pd.Series(bearish_level)
    bullish_level_series = pd.Series(bullish_level)
    bearish_level_ffill = bearish_level_series.ffill().values
    bullish_level_ffill = bullish_level_series.ffill().values
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d data to 12h timeframe
    # Williams Fractals need 2 extra bars for confirmation (as per Williams theory)
    bearish_level_aligned = align_htf_to_ltf(prices, df_1d, bearish_level_ffill, additional_delay_bars=2)
    bullish_level_aligned = align_htf_to_ltf(prices, df_1d, bullish_level_ffill, additional_delay_bars=2)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike (2x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA34 has enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_level_aligned[i]) or np.isnan(bullish_level_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above bearish fractal level (resistance) with 1d uptrend and volume spike
            if (close[i] > bearish_level_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below bullish fractal level (support) with 1d downtrend and volume spike
            elif (close[i] < bullish_level_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below bullish fractal level (support) or trend fails
            if (close[i] < bullish_level_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above bearish fractal level (resistance) or trend fails
            if (close[i] > bearish_level_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals