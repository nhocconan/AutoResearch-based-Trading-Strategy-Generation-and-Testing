#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation
# - Williams Fractals from 6h: bullish fractal breakout = long, bearish fractal breakout = short
# - 1d ADX(14) > 25 to ensure trending market and avoid chop
# - Volume confirmation: current 6h volume > 2.0x 20-period average
# - ATR-based trailing stop: exit long when price < highest_high - 2.5*ATR, exit short when price > lowest_low + 2.5*ATR
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Williams Fractals provide high-probability reversal/continuation signals
# - Works in bull/bear markets: ADX filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "6h_1d_williams_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h Williams Fractals
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Williams Fractals: 5-bar pattern
    # Bullish fractal: low[i-2] > low[i] and low[i-1] > low[i] and low[i+1] > low[i] and low[i+2] > low[i]
    # Bearish fractal: high[i-2] < high[i] and high[i-1] < high[i] and high[i+1] < high[i] and high[i+2] < high[i]
    bullish_fractal = np.zeros(n, dtype=bool)
    bearish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (low_6h[i-2] > low_6h[i] and low_6h[i-1] > low_6h[i] and 
            low_6h[i+1] > low_6h[i] and low_6h[i+2] > low_6h[i]):
            bullish_fractal[i] = True
        if (high_6h[i-2] < high_6h[i] and high_6h[i-1] < high_6h[i] and 
            high_6h[i+1] < high_6h[i] and high_6h[i+2] < high_6h[i]):
            bearish_fractal[i] = True
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_20)
    
    # Pre-compute 6h ATR(14) for trailing stop
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_14 = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0  # for trailing stop
    lowest_low = 0.0    # for trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high for trailing stop
            if close_6h[i] > highest_high:
                highest_high = close_6h[i]
            # Exit: trailing stop hit
            if close_6h[i] < highest_high - 2.5 * atr_14[i]:
                position = 0
                highest_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            if close_6h[i] < lowest_low:
                lowest_low = close_6h[i]
            # Exit: trailing stop hit
            if close_6h[i] > lowest_low + 2.5 * atr_14[i]:
                position = 0
                lowest_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams Fractal breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Breakout long: bullish fractal formed and price breaks above it
                if bullish_fractal[i] and close_6h[i] > high_6h[i]:
                    position = 1
                    entry_price = close_6h[i]
                    highest_high = close_6h[i]
                    signals[i] = 0.25
                # Breakout short: bearish fractal formed and price breaks below it
                elif bearish_fractal[i] and close_6h[i] < low_6h[i]:
                    position = -1
                    entry_price = close_6h[i]
                    lowest_low = close_6h[i]
                    signals[i] = -0.25
    
    return signals