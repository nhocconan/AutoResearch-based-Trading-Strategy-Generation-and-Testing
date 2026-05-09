#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with weekly trend filter and volume confirmation.
# Uses Williams Fractal (bearish for short, bullish for long) as reversal signals.
# Weekly EMA200 filters for long-term trend direction to avoid counter-trend trades.
# Volume > 1.3x 20-period EMA ensures institutional participation.
# Designed for 12h timeframe to capture multi-day swings with low trade frequency.
name = "12h_WilliamsFractal_Breakout_WeeklyEMA200_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily data
    # Bearish fractal: high[n] is highest of 5 bars (n-2, n-1, n, n+1, n+2)
    # Bullish fractal: low[n] is lowest of 5 bars (n-2, n-1, n, n+1, n+2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Williams Fractals need 2 extra bars for confirmation (bar closes after fractal)
    bearish_fractal_confirmed = np.where(bearish_fractal, 1.0, np.nan)
    bullish_fractal_confirmed = np.where(bullish_fractal, 1.0, np.nan)
    
    # Align fractals to 12h with 2-bar confirmation delay
    bearish_12h = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirmed, additional_delay_bars=2)
    bullish_12h = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirmed, additional_delay_bars=2)
    
    # Weekly EMA200 trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume spike filter: volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.3 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure weekly EMA200 is ready
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(bearish_12h[i]) or np.isnan(bullish_12h[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: bullish fractal breakout with volume spike and above weekly EMA200
            if (bullish_12h[i] == 1.0 and vol_spike[i] and price > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal breakout with volume spike and below weekly EMA200
            elif (bearish_12h[i] == 1.0 and vol_spike[i] and price < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish fractal appears (potential top)
            if bearish_12h[i] == 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish fractal appears (potential bottom)
            if bullish_12h[i] == 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals