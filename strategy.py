#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h EMA50 trend filter and volume confirmation
# Uses Williams fractals from 12h chart to identify key swing points.
# Enters long when price breaks above the most recent bullish fractal with volume confirmation and 12h EMA50 uptrend.
# Enters short when price breaks below the most recent bearish fractal with volume confirmation and 12h EMA50 downtrend.
# Williams fractals provide structure, volume confirms breakout validity, EMA50 filters trend.
# Designed for 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via breakdowns.

name = "6h_WilliamsFractal_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams fractal calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Williams fractals on 12h data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal = np.full(len(high_12h), np.nan)
    bullish_fractal = np.full(len(low_12h), np.nan)
    
    for i in range(2, len(high_12h) - 2):
        if (high_12h[i-2] < high_12h[i-1] and 
            high_12h[i] < high_12h[i-1] and
            high_12h[i-3] < high_12h[i-1] and
            high_12h[i+1] < high_12h[i-1]):
            bearish_fractal[i-1] = high_12h[i-1]
        
        if (low_12h[i-2] > low_12h[i-1] and 
            low_12h[i] > low_12h[i-1] and
            low_12h[i-3] > low_12h[i-1] and
            low_12h[i+1] > low_12h[i-1]):
            bullish_fractal[i-1] = low_12h[i-1]
    
    # Align Williams fractals to 6h timeframe with additional delay for confirmation
    # Williams fractals need 2 extra 12h bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # Get 12h data for EMA50 trend filter - ONCE before loop
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe (wait for completed 12h bar)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above most recent bullish fractal AND volume spike AND 12h EMA50 uptrend
            if (close[i] > bullish_fractal_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below most recent bearish fractal AND volume spike AND 12h EMA50 downtrend
            elif (close[i] < bearish_fractal_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters below the bullish fractal OR trend reverses
            if (close[i] <= bullish_fractal_aligned[i]) or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above the bearish fractal OR trend reverses
            if (close[i] >= bearish_fractal_aligned[i]) or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals