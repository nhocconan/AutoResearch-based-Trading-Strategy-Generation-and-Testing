#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with 12h trend filter and volume confirmation.
# Uses 6h timeframe to balance trade frequency and responsiveness.
# Williams Fractals identify potential reversal points; breakouts above/below fractals with volume
# indicate strong momentum. 12h trend filter ensures alignment with higher timeframe direction.
# Volume confirmation reduces false breakouts. Designed to work in both bull and bear markets
# by following 12h trend direction (long in uptrend, short in downtrend).
name = "6h_WilliamsFractal_Breakout_12hTrend_Volume"
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
    
    # 12h data for trend filter and Williams Fractals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Fractals: bearish (high) and bullish (low) fractals
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n_12h = len(high_12h)
    bearish_fractal = np.zeros(n_12h, dtype=bool)
    bullish_fractal = np.zeros(n_12h, dtype=bool)
    
    for i in range(2, n_12h - 2):
        # Bearish fractal: middle bar highest
        if (high_12h[i-2] < high_12h[i-1] and 
            high_12h[i] < high_12h[i-1] and
            high_12h[i-3] < high_12h[i-1] and
            high_12h[i+1] < high_12h[i-1]):
            bearish_fractal[i-1] = True
        
        # Bullish fractal: middle bar lowest
        if (low_12h[i-2] > low_12h[i-1] and 
            low_12h[i] > low_12h[i-1] and
            low_12h[i-3] > low_12h[i-1] and
            low_12h[i+1] > low_12h[i-1]):
            bullish_fractal[i-1] = True
    
    # Convert to price levels: fractal high/low values
    bearish_level = np.where(bearish_fractal, high_12h, np.nan)
    bullish_level = np.where(bullish_fractal, low_12h, np.nan)
    
    # Forward fill to get the most recent fractal level
    bearish_level_series = pd.Series(bearish_level)
    bullish_level_series = pd.Series(bullish_level)
    bearish_level_ff = bearish_level_series.ffill().values
    bullish_level_ff = bullish_level_series.ffill().values
    
    # Need 2-bar confirmation for fractals (as per Williams)
    bearish_level_conf = bearish_level_ff.copy()
    bullish_level_conf = bullish_level_ff.copy()
    # Apply additional 2-bar delay for confirmation
    bearish_level_conf = np.roll(bearish_level_conf, 2)
    bullish_level_conf = np.roll(bullish_level_conf, 2)
    bearish_level_conf[:2] = np.nan
    bullish_level_conf[:2] = np.nan
    
    # Align to 6h timeframe with 2-bar confirmation delay
    bearish_6h = align_htf_to_ltf(prices, df_12h, bearish_level_conf, additional_delay_bars=2)
    bullish_6h = align_htf_to_ltf(prices, df_12h, bullish_level_conf, additional_delay_bars=2)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike filter: volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(bearish_6h[i]) or np.isnan(bullish_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal (support) with volume spike and above 12h EMA50 (uptrend)
            if (price > bullish_6h[i] and vol_spike[i] and price > ema_50_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal (resistance) with volume spike and below 12h EMA50 (downtrend)
            elif (price < bearish_6h[i] and vol_spike[i] and price < ema_50_6h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below bearish fractal or volume dries up
            if (price < bearish_6h[i]) or (not vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above bullish fractal or volume dries up
            if (price > bullish_6h[i]) or (not vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals