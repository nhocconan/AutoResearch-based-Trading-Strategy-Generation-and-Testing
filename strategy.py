#!/usr/bin/env python3
"""
1d_Williams_Fractal_Reversal_With_Weekly_Trend
Strategy: Long when bullish fractal forms during weekly uptrend with volume confirmation.
Short when bearish fractal forms during weekly downtrend with volume confirmation.
Exit when opposite fractal forms or trend weakens.
Position size: 0.25
Designed to capture reversals at key support/resistance with trend alignment.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly trend using EMA34
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_ltf_to_htf(prices, df_1w, ema34_1w)
    
    # Calculate Williams fractals on daily data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    
    # We'll calculate these manually since we need to look at patterns
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        # Bearish fractal: middle high is highest of 5
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = True
        
        # Bullish fractal: middle low is lowest of 5
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = True
    
    # Align fractals (they are already on daily timeframe, so no alignment needed for 1d)
    # But we need to ensure we only use completed fractals (no look-ahead)
    # Since we calculated using i-2 to i+2, we need to shift by 2 to avoid look-ahead
    bearish_fractal_aligned = np.roll(bearish_fractal, 2)
    bullish_fractal_aligned = np.roll(bullish_fractal, 2)
    # Set first 2 values to False to avoid wraparound issues
    bearish_fractal_aligned[:2] = False
    bullish_fractal_aligned[:2] = False
    
    # Volume confirmation (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: weekly EMA34 slope
        if i >= 35:
            ema34_slope = ema34_1w_aligned[i] - ema34_1w_aligned[i-1]
            uptrend = ema34_slope > 0
            downtrend = ema34_slope < 0
        else:
            uptrend = False
            downtrend = False
        
        if position == 0:
            # Long: bullish fractal + weekly uptrend + volume confirmation
            if bullish_fractal_aligned[i] and uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal + weekly downtrend + volume confirmation
            elif bearish_fractal_aligned[i] and downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish fractal forms or trend turns down
            if bearish_fractal_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish fractal forms or trend turns up
            if bullish_fractal_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Fractal_Reversal_With_Weekly_Trend"
timeframe = "1d"
leverage = 1.0