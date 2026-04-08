#!/usr/bin/env python3
# 1d_fractal_breakout_weekly_trend_volume_v1
# Hypothesis: On daily timeframe, price breaking above/below weekly Williams fractal levels with volume surge (>2x average) and aligned weekly trend (EMA13) provides high-probability continuation trades. Works in bull/bear by following weekly trend. Target: 10-20 trades/year via strict fractal breaks + volume + trend alignment.

name = "1d_fractal_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend and fractal levels
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly EMA13 for trend filter
    ema13_w = pd.Series(close_w).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Weekly Williams fractals (requires 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_w, low_w)
    
    # Align weekly data to daily with 2-bar confirmation for fractals
    ema13_w_aligned = align_htf_to_ltf(prices, df_w, ema13_w)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_w, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_w, bearish_fractal, additional_delay_bars=2)
    
    # 20-day average volume for volume surge
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0  # Track bars since entry for minimum holding period
    
    # Start from sufficient lookback
    start_idx = max(20, 13)  # Need volume MA and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_w_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        if position == 1:  # Long position
            bars_since_entry += 1
            # Exit if price breaks below weekly bullish fractal OR max 10 days held
            if close[i] < bullish_fractal_aligned[i] or bars_since_entry >= 10:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit if price breaks above weekly bearish fractal OR max 10 days held
            if close[i] > bearish_fractal_aligned[i] or bars_since_entry >= 10:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            bars_since_entry = 0
            # Volume surge: current volume > 2x 20-day average
            volume_surge = volume[i] > 2.0 * vol_ma[i]
            
            # Breakout long: price crosses above weekly bullish fractal with volume surge and weekly uptrend
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i-1] <= bullish_fractal_aligned[i-1] and  # crossed above this bar
                volume_surge and 
                ema13_w_aligned[i] > ema13_w_aligned[i-1]):  # weekly EMA13 rising = uptrend
                position = 1
                signals[i] = 0.25
            # Breakout short: price crosses below weekly bearish fractal with volume surge and weekly downtrend
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i-1] >= bearish_fractal_aligned[i-1] and  # crossed below this bar
                  volume_surge and 
                  ema13_w_aligned[i] < ema13_w_aligned[i-1]):  # weekly EMA13 falling = downtrend
                position = -1
                signals[i] = -0.25
    
    return signals