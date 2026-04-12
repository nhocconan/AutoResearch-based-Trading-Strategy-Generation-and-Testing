#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for fractal detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Williams fractals on weekly data
    # Bearish fractal: high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    bearish = np.zeros(len(high_1w), dtype=bool)
    bullish = np.zeros(len(low_1w), dtype=bool)
    
    for i in range(2, len(high_1w) - 2):
        if (high_1w[i] > high_1w[i-2] and high_1w[i] > high_1w[i-1] and 
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish[i] = True
        if (low_1w[i] < low_1w[i-2] and low_1w[i] < low_1w[i-1] and 
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish[i] = True
    
    # Convert to price levels (use the fractal high/low values)
    bearish_level = np.where(bearish, high_1w, np.nan)
    bullish_level = np.where(bullish, low_1w, np.nan)
    
    # Forward fill to get the most recent fractal level
    bearish_series = pd.Series(bearish_level)
    bullish_series = pd.Series(bullish_level)
    bearish_ff = bearish_series.ffill().values
    bullish_ff = bullish_series.ffill().values
    
    # Align to 6h timeframe with 2-bar delay for confirmation (fractals need 2 future candles to confirm)
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_ff, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_ff, additional_delay_bars=2)
    
    # Volume filter - 50-period average on 6h data (longer for stability)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above recent bearish fractal with volume (breakout of resistance)
        long_signal = close[i] > bearish_aligned[i] and volume_ok[i]
        # Short: price breaks below recent bullish fractal with volume (breakdown of support)
        short_signal = close[i] < bullish_aligned[i] and volume_ok[i]
        
        # Exit when price crosses the opposite fractal level
        exit_long = close[i] < bullish_aligned[i]  # Price breaks below bullish fractal (support)
        exit_short = close[i] > bearish_aligned[i]  # Price breaks above bearish fractal (resistance)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals