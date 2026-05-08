#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation.
# Long when price breaks above weekly bullish fractal (resistance) AND weekly EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below weekly bearish fractal (support) AND weekly EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the weekly fractal range (between bullish and bearish fractal).
# Williams Fractals identify key swing points with natural lag, reducing false breakouts.
# Weekly EMA50 filters higher timeframe trend to avoid counter-trend trades.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsFractal_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for fractal calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Williams Fractals from weekly OHLC
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and low[n-2] < low[n-1] > low[n]
    # Bullish fractal: high[n-2] > high[n-1] < high[n] and low[n-2] > low[n-1] < low[n]
    high_vals = df_1w['high'].values
    low_vals = df_1w['low'].values
    
    bearish_fractal = np.full_like(high_vals, np.nan)
    bullish_fractal = np.full_like(low_vals, np.nan)
    
    # Need at least 5 points for fractal calculation (2 on each side)
    for i in range(2, len(high_vals) - 2):
        # Bearish fractal (peak)
        if (high_vals[i-2] < high_vals[i-1] and 
            high_vals[i] < high_vals[i-1] and
            low_vals[i-2] < low_vals[i-1] and
            low_vals[i] < low_vals[i-1]):
            bearish_fractal[i-1] = high_vals[i-1]
        
        # Bullish fractal (trough)
        if (high_vals[i-2] > high_vals[i-1] and 
            high_vals[i] > high_vals[i-1] and
            low_vals[i-2] > low_vals[i-1] and
            low_vals[i] > low_vals[i-1]):
            bullish_fractal[i-1] = low_vals[i-1]
    
    # Williams Fractals need 2 extra weekly bars for confirmation (as per rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly EMA50 direction
    ema50_rising = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1w_aligned[1:] > ema50_1w_aligned[:-1]
    ema50_falling[1:] = ema50_1w_aligned[1:] < ema50_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 2)  # Sufficient warmup for EMA50 and fractals
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above bullish fractal, weekly EMA50 rising, volume filter
            long_cond = (close[i] > bullish_fractal_aligned[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below bearish fractal, weekly EMA50 falling, volume filter
            short_cond = (close[i] < bearish_fractal_aligned[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below bearish fractal (weekly support)
            if close[i] < bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above bullish fractal (weekly resistance)
            if close[i] > bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals