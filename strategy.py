#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
# Williams fractals identify potential turning points. Breakouts from fractal levels
# with 1d trend alignment capture momentum moves. Volume filters reduce false breakouts.
# Works in bull/bear by following 1d trend direction only.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsFractal_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals, trend, and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Williams Fractals: bearish (high) and bullish (low)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    high_arr = df_1d['high'].values
    low_arr = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_arr), np.nan)
    bullish_fractal = np.full(len(low_arr), np.nan)
    
    for i in range(2, len(high_arr) - 2):
        if (high_arr[i-2] < high_arr[i-1] and 
            high_arr[i] > high_arr[i-1] and 
            high_arr[i] > high_arr[i+1] and 
            high_arr[i] > high_arr[i+2]):
            bearish_fractal[i] = high_arr[i]  # Store the fractal high level
        
        if (low_arr[i-2] > low_arr[i-1] and 
            low_arr[i] < low_arr[i-1] and 
            low_arr[i] < low_arr[i+1] and 
            low_arr[i] < low_arr[i+2]):
            bullish_fractal[i] = low_arr[i]   # Store the fractal low level
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 1d volume > 1.3 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.3)
    
    # Williams fractals need 2 extra bars for confirmation (after the center bar forms)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20) + 2  # Need EMA50, volume MA, and fractal confirmation
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bearish_level = bearish_fractal_aligned[i]  # Resistance from bearish fractal
        bullish_level = bullish_fractal_aligned[i]   # Support from bullish fractal
        trend = ema50_1d_aligned[i]
        vol_filter = volume_filter_1d_aligned[i]
        
        if position == 0:
            # Enter long: break above bullish fractal (support turned resistance?) 
            # Actually: bullish fractal is a support level, break above it with trend up
            if close[i] > bullish_level and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below bearish fractal (resistance level) with trend down
            elif close[i] < bearish_level and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below the bullish fractal level (support)
            if close[i] <= bullish_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above the bearish fractal level (resistance)
            if close[i] >= bearish_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals