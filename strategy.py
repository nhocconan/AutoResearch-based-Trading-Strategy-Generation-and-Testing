#!/usr/bin/env python3
"""
4h_1d_Williams_Fractal_Breakout_Trend_Filter
Hypothesis: 4h Williams Fractal breakout with 1d trend filter. Williams Fractals identify key reversal points where price respects support/resistance. The 1d EMA200 provides trend direction to avoid counter-trend trades. This combination should work in both bull and bear markets by filtering breakouts with the higher timeframe trend. Designed for 4h to target 75-200 total trades over 4 years (19-50/year).
"""

name = "4h_1d_Williams_Fractal_Breakout_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 4h data
    def williams_fractals(high, low):
        n = len(high)
        bullish = np.zeros(n, dtype=bool)
        bearish = np.zeros(n, dtype=bool)
        for i in range(2, n-2):
            if (high[i] > high[i-1] and high[i] > high[i-2] and 
                high[i] > high[i+1] and high[i] > high[i+2]):
                bearish[i] = True
            if (low[i] < low[i-1] and low[i] < low[i-2] and 
                low[i] < low[i+1] and low[i] < low[i+2]):
                bullish[i] = True
        return bullish, bearish
    
    bullish_fractal, bearish_fractal = williams_fractals(high, low)
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate EMA200 on daily close
    close_1d = df_1d['close'].values
    ema_200 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        multiplier = 2 / (200 + 1)
        ema_200[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200[i] = close_1d[i] * multiplier + ema_200[i-1] * (1 - multiplier)
    
    # Align EMA200 to 4h timeframe (with 1-bar delay for confirmation)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA200 for long, below for short
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        if position == 0:
            # Long: bullish fractal break above recent high with volume and uptrend
            # Look for recent bullish fractal and break above its high
            if bullish_fractal[i]:
                # Find the high at the fractal point
                fractal_high = high[i]
                if (close[i] > fractal_high and 
                    volume_confirm[i] and 
                    uptrend):
                    signals[i] = 0.25
                    position = 1
            # Short: bearish fractal break below recent low with volume and downtrend
            elif bearish_fractal[i]:
                fractal_low = low[i]
                if (close[i] < fractal_low and 
                    volume_confirm[i] and 
                    downtrend):
                    signals[i] = -0.25
                    position = -1
                    
        elif position == 1:
            # Long: exit on bearish fractal or trend change
            if bearish_fractal[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit on bullish fractal or trend change
            if bullish_fractal[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals