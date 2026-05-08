#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above a bearish Williams fractal (resistance) AND 1w EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below a bullish Williams fractal (support) AND 1w EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the last fractal level (between support and resistance).
# Williams fractals identify key support/resistance levels. The 1w EMA50 filter ensures we trade with the higher timeframe trend.
# Volume spike confirms institutional participation. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_WilliamsFractal_Breakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w EMA50 direction
    ema50_rising = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1w_aligned[1:] > ema50_1w_aligned[:-1]
    ema50_falling[1:] = ema50_1w_aligned[1:] < ema50_1w_aligned[:-1]
    
    # 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Williams fractals: bearish (resistance) and bullish (support)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    # Williams fractal calculation: need 5 points (2 left, 2 right)
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: high is highest of 5 points
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        # Bullish fractal: low is lowest of 5 points
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals need 2 extra bars for confirmation (after the center bar)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    last_resistance = np.nan  # Last bearish fractal (resistance) level
    last_support = np.nan     # Last bullish fractal (support) level
    
    start_idx = max(50, 10)  # Sufficient warmup for EMA50 and fractals
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Update fractal levels if new fractal formed
        if not np.isnan(bearish_fractal_aligned[i]):
            last_resistance = bearish_fractal_aligned[i]
        if not np.isnan(bullish_fractal_aligned[i]):
            last_support = bullish_fractal_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above last resistance, 1w EMA50 rising, volume filter
            long_cond = (not np.isnan(last_resistance) and 
                        close[i] > last_resistance and 
                        ema50_rising[i] and 
                        volume_filter[i])
            # Short conditions: price breaks below last support, 1w EMA50 falling, volume filter
            short_cond = (not np.isnan(last_support) and 
                         close[i] < last_support and 
                         ema50_falling[i] and 
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below last support
            if not np.isnan(last_support) and close[i] < last_support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above last resistance
            if not np.isnan(last_resistance) and close[i] > last_resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals