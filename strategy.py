#!/usr/bin/env python3
# 1d_weekly_fractal_breakout_volume_v3
# Hypothesis: 1d timeframe trading using weekly Williams Fractal breakouts with volume confirmation. Fractals provide key support/resistance levels; breakouts with volume capture momentum in both bull and bear markets. Weekly trend filter ensures alignment with higher timeframe direction. Target: 10-30 trades/year per symbol.

name = "1d_weekly_fractal_breakout_volume_v3"
timeframe = "1d"
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
    
    # Get weekly data for fractals and trend filter - call ONCE before loop
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_w = pd.Series(close_w).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate weekly Williams Fractals
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_w, low_w)
    # Need 2-bar confirmation for fractals (wait for 2 candles after the fractal)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_w, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 20-period average volume for 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned weekly indicators for current 1d bar
        ema20_val = align_htf_to_ltf(prices, df_w, ema20_w)[i]
        bearish_val = bearish_fractal_aligned[i]
        bullish_val = bullish_fractal_aligned[i]
        
        # Skip if any required data is NaN
        if np.isnan(ema20_val) or np.isnan(vol_ma[i]) or volume[i] == 0:
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 2.0x 20-period average (stricter for fewer trades)
        vol_breakout = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema20_val
        downtrend = close[i] < ema20_val
        
        if position == 1:  # Long position
            # Exit if price breaks below bullish fractal (support)
            if not np.isnan(bullish_val) and close[i] < bullish_val:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above bearish fractal (resistance)
            if not np.isnan(bearish_val) and close[i] > bearish_val:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above bearish fractal (resistance) with volume confirmation and uptrend
            if not np.isnan(bearish_val) and high[i] >= bearish_val and close[i] > bearish_val and vol_breakout and uptrend:
                position = 1
                signals[i] = 0.25
            # Breakout short below bullish fractal (support) with volume confirmation and downtrend
            elif not np.isnan(bullish_val) and low[i] <= bullish_val and close[i] < bullish_val and vol_breakout and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals