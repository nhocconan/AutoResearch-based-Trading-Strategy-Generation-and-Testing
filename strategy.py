#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal Breakout with Daily Trend Filter
# - Bearish fractal (sell signal) + price below daily EMA200 = short
# - Bullish fractal (buy signal) + price above daily EMA200 = long
# - Williams fractals provide high-probability reversal signals at swing points
# - Daily EMA200 filter ensures alignment with long-term trend
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data for fractal calculation and EMA200
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams fractals on daily timeframe
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Calculate daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Williams fractals need 2 extra daily bars for confirmation (right side of fractal)
    bearish_fractal_12h = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_12h = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema200_12h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 12h EMA50 for entry timing
    close_12h = prices['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in indicators
        if np.isnan(bearish_fractal_12h[i]) or np.isnan(bullish_fractal_12h[i]) or \
           np.isnan(ema200_12h[i]) or np.isnan(ema50_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        ema200 = ema200_12h[i]
        ema50 = ema50_12h[i]
        
        if position == 0:
            # Long entry: Bullish fractal + price above daily EMA200 + price above EMA50
            if bullish_fractal_12h[i] and price > ema200 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish fractal + price below daily EMA200 + price below EMA50
            elif bearish_fractal_12h[i] and price < ema200 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below EMA50 or bearish fractal appears
            if price < ema50 or bearish_fractal_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above EMA50 or bullish fractal appears
            if price > ema50 or bullish_fractal_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_EMA200Filter"
timeframe = "12h"
leverage = 1.0