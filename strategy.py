#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Williams Fractal breakout and 12h EMA trend filter.
# Uses daily Williams fractals to identify potential breakout levels, with 12h EMA for trend direction.
# Works in both bull and bear markets by only taking breakouts in the direction of the higher timeframe trend.
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and cost.
name = "6h_12hEMA_1dWilliamsFractalBreakout"
timeframe = "6h"
leverage = 1.0

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (high) and bullish (low)"""
    n = len(high)
    bearish = np.zeros(n, dtype=bool)
    bullish = np.zeros(n, dtype=bool)
    
    for i in range(2, n - 2):
        # Bearish fractal: highest high with two lower highs on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = True
        # Bullish fractal: lowest low with two higher lows on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = True
            
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams fractals on 1d timeframe
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    
    # Williams fractals need 2 extra bars for confirmation (formation + confirmation)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Get 12h data for EMA trend filter (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for EMA and fractals
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Check for fractal breakouts
        bullish_breakout = bullish_fractal_confirmed[i] > 0.5 and close[i] > high[i-1]
        bearish_breakout = bearish_fractal_confirmed[i] > 0.5 and close[i] < low[i-1]
        
        if position == 0:
            # Long when bullish fractal breakout and price above 12h EMA34
            if bullish_breakout and close[i] > ema34_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when bearish fractal breakout and price below 12h EMA34
            elif bearish_breakout and close[i] < ema34_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price closes below 12h EMA34 or bearish fractal breakout
            if close[i] < ema34_12h_aligned[i] or bearish_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price closes above 12h EMA34 or bullish fractal breakout
            if close[i] > ema34_12h_aligned[i] or bullish_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals