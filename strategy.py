#!/usr/bin/env python3
# 4h_fractal_breakout_1d_trend_volume_v5
# Hypothesis: 4h timeframe trading using daily Williams Fractal breakouts with volume confirmation and 1d trend filter. Fractals provide key support/resistance levels; breakouts with volume capture momentum in both bull and bear markets. Daily trend filter ensures alignment with higher timeframe direction. Tightened volume threshold (3.0x) and reduced position size (0.20) to limit trades to target range (19-50/year).

name = "4h_fractal_breakout_1d_trend_volume_v5"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (up) and bullish (down) fractals."""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest of 5 bars
        if high[i] >= high[i-1] and high[i] >= high[i-2] and high[i] >= high[i+1] and high[i] >= high[i+2]:
            bearish[i] = high[i]
        # Bullish fractal: low[i] is lowest of 5 bars
        if low[i] <= low[i-1] and low[i] <= low[i-2] and low[i] <= low[i+1] and low[i] <= low[i+2]:
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for fractals and trend filter - call ONCE before loop
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Calculate daily EMA20 for trend filter
    ema20_d = pd.Series(close_d).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate daily Williams Fractals
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_d, low_d)
    # Need 2-bar confirmation for fractals (wait for 2 candles after the fractal)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 20-period average volume for 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned daily indicators for current 4h bar
        ema20_val = align_htf_to_ltf(prices, df_d, ema20_d)[i]
        bearish_val = bearish_fractal_aligned[i]
        bullish_val = bullish_fractal_aligned[i]
        
        # Skip if any required data is NaN
        if np.isnan(ema20_val) or np.isnan(vol_ma[i]) or volume[i] == 0:
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 3.0x 20-period average (stricter for fewer trades)
        vol_breakout = volume[i] > 3.0 * vol_ma[i]
        
        # Trend filter: price above/below daily EMA20
        uptrend = close[i] > ema20_val
        downtrend = close[i] < ema20_val
        
        if position == 1:  # Long position
            # Exit if price breaks below bullish fractal (support)
            if not np.isnan(bullish_val) and close[i] < bullish_val:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit if price breaks above bearish fractal (resistance)
            if not np.isnan(bearish_val) and close[i] > bearish_val:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Breakout long above bearish fractal (resistance) with volume confirmation and uptrend
            if not np.isnan(bearish_val) and high[i] >= bearish_val and close[i] > bearish_val and vol_breakout and uptrend:
                position = 1
                signals[i] = 0.20
            # Breakout short below bullish fractal (support) with volume confirmation and downtrend
            elif not np.isnan(bullish_val) and low[i] <= bullish_val and close[i] < bullish_val and vol_breakout and downtrend:
                position = -1
                signals[i] = -0.20
    
    return signals