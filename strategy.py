#!/usr/bin/env python3
"""
6h_WilliamsFractal_1dTrend_Breakout
Hypothesis: Williams fractals identify key swing points on daily chart. 
Breakout above/below recent fractal with 1d trend filter and volume confirmation.
In bull markets, captures upside breaks above bullish fractals; in bear markets, captures downside breaks below bearish fractals.
Designed for ~20-40 trades/year to minimize fee drag while capturing high-probability directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams fractal detection (5-bar pattern: high[2] > high[1] & high[0] and high[2] > high[3] & high[4])
    # Bearish fractal: highest high in middle with lower highs on both sides
    # Bullish fractal: lowest low in middle with higher lows on both sides
    n1 = len(high)
    bearish = np.zeros(n1, dtype=bool)
    bullish = np.zeros(n1, dtype=bool)
    
    for i in range(2, n1 - 2):
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = True
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = True
    
    # Get 1d data for fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate fractals on 1d data
    n_1d = len(high_1d)
    bearish_1d = np.zeros(n_1d, dtype=bool)
    bullish_1d = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_1d[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_1d[i] = True
    
    # Trend filter: 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_1d.astype(float))
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d.astype(float))
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bearish_1d_aligned[i]) or
            np.isnan(bullish_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_1d_aligned[i]
        bearish_fractal = bearish_1d_aligned[i] > 0.5
        bullish_fractal = bullish_1d_aligned[i] > 0.5
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal level with volume spike and uptrend
            # Find most recent bullish fractal level
            if bullish_fractal and i >= 2:
                # Look back for the fractal level (low of the fractal candle)
                j = i
                while j >= max(0, i - 50):  # Look back max 50 bars
                    if bullish_1d_aligned[j] > 0.5:
                        fractal_level = low[j]  # Use the low of the bullish fractal candle
                        break
                    j -= 1
                else:
                    fractal_level = price
                
                if price > fractal_level and vol_spike and price > ema34:
                    signals[i] = 0.25
                    position = 1
            
            # Short: price breaks below bearish fractal level with volume spike and downtrend
            elif bearish_fractal and i >= 2:
                # Look back for the fractal level (high of the fractal candle)
                j = i
                while j >= max(0, i - 50):  # Look back max 50 bars
                    if bearish_1d_aligned[j] > 0.5:
                        fractal_level = high[j]  # Use the high of the bearish fractal candle
                        break
                    j -= 1
                else:
                    fractal_level = price
                
                if price < fractal_level and vol_spike and price < ema34:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below EMA34 OR breaks below recent bullish fractal
            if price < ema34:
                signals[i] = 0.0
                position = 0
            elif bullish_fractal and i >= 2:
                j = i
                while j >= max(0, i - 50):
                    if bullish_1d_aligned[j] > 0.5:
                        if price < low[j]:
                            signals[i] = 0.0
                            position = 0
                        break
                    j -= 1
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above EMA34 OR breaks above recent bearish fractal
            if price > ema34:
                signals[i] = 0.0
                position = 0
            elif bearish_fractal and i >= 2:
                j = i
                while j >= max(0, i - 50):
                    if bearish_1d_aligned[j] > 0.5:
                        if price > high[j]:
                            signals[i] = 0.0
                            position = 0
                        break
                    j -= 1
    
    return signals

name = "6h_WilliamsFractal_1dTrend_Breakout"
timeframe = "6h"
leverage = 1.0