#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
# Uses Williams Fractals to identify key support/resistance levels from 1d timeframe.
# Breakouts above bearish fractals (resistance) or below bullish fractals (support) 
# with volume confirmation and aligned with 1d EMA50 trend capture strong momentum.
# Works in both bull and bear markets by following higher timeframe trend.
name = "6h_WilliamsFractal_Breakout_1dEMA50_Volume"
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
    
    # Daily data for Williams Fractals and EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    # Williams Fractal: need 5 points (2 left, 2 right)
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: high[i] is highest among 5 points
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: low[i] is lowest among 5 points
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams Fractals need 2-bar confirmation after the center bar
    bearish_fractal_confirmed = np.roll(bearish_fractal, 2)
    bullish_fractal_confirmed = np.roll(bullish_fractal, 2)
    bearish_fractal_confirmed[:2] = np.nan
    bullish_fractal_confirmed[:2] = np.nan
    
    # Align fractals to 6h timeframe
    bearish_fractal_6h = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirmed, additional_delay_bars=0)
    bullish_fractal_6h = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirmed, additional_delay_bars=0)
    
    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(bearish_fractal_6h[i]) or np.isnan(bullish_fractal_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance) with volume spike and above daily EMA50
            if (price > bearish_fractal_6h[i] and vol_spike[i] and price > ema_50_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal (support) with volume spike and below daily EMA50
            elif (price < bullish_fractal_6h[i] and vol_spike[i] and price < ema_50_6h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below bullish fractal (support)
            if price < bullish_fractal_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above bearish fractal (resistance)
            if price > bearish_fractal_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals