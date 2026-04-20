#!/usr/bin/env python3
"""
6h_WilliamsFractal_1d_Breakout
Hypothesis: Williams Fractal breakout on 6h with 1d trend filter. Buy when price breaks above latest bearish fractal resistance in uptrend, sell when breaks below latest bullish fractal support in downtrend. Works in bull/bear by following trend. Uses fractals for natural support/resistance, reducing whipsaw vs fixed periods.
Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25.
"""

name = "6h_WilliamsFractal_1d_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: using align_ltf_to_htf for HTF values that need no delay

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    def calculate_ema(values, period):
        ema = np.full_like(values, np.nan)
        if len(values) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema34_1d = calculate_ema(close_1d, 34)
    ema34_1d_aligned = align_ltf_to_htf(prices, df_1d, ema34_1d)  # HTF trend, no extra delay needed
    
    # Williams Fractals: 5-bar pattern (high[2] > high[1] & high[3] and high[2] > high[0] & high[4])
    def calculate_williams_fractals(high, low):
        n = len(high)
        bearish = np.zeros(n)  # resistance fractal (peak)
        bullish = np.zeros(n)  # support fractal (trough)
        
        for i in range(2, n-2):
            # Bearish fractal: highest high in middle
            if (high[i] > high[i-1] and high[i] > high[i+1] and 
                high[i] > high[i-2] and high[i] > high[i+2]):
                bearish[i] = high[i]
            
            # Bullish fractal: lowest low in middle
            if (low[i] < low[i-1] and low[i] < low[i+1] and 
                low[i] < low[i-2] and low[i] < low[i+2]):
                bullish[i] = low[i]
        
        return bearish, bullish
    
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    # Fractals need 2-bar confirmation (wait for 2 candles after the fractal bar)
    bearish_fractal_aligned = align_ltf_to_htf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_ltf_to_htf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Track latest fractal levels (forward fill)
    latest_bearish = np.full(n, np.nan)
    latest_bullish = np.full(n, np.nan)
    
    last_bearish = np.nan
    last_bullish = np.nan
    for i in range(n):
        if not np.isnan(bearish_fractal_aligned[i]):
            last_bearish = bearish_fractal_aligned[i]
        if not np.isnan(bullish_fractal_aligned[i]):
            last_bullish = bullish_fractal_aligned[i]
        latest_bearish[i] = last_bearish
        latest_bullish[i] = last_bullish
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(latest_bearish[i]) or 
            np.isnan(latest_bullish[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Uptrend: close > EMA34
            # Long: price breaks above bearish fractal (resistance)
            if close[i] > ema34_1d_aligned[i] and close[i] > latest_bearish[i]:
                signals[i] = 0.25
                position = 1
            # Downtrend: close < EMA34
            # Short: price breaks below bullish fractal (support)
            elif close[i] < ema34_1d_aligned[i] and close[i] < latest_bullish[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change OR price breaks below bullish fractal (support)
            if close[i] < ema34_1d_aligned[i] or close[i] < latest_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change OR price breaks above bearish fractal (resistance)
            if close[i] > ema34_1d_aligned[i] or close[i] > latest_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals