#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal + Weekly Trend + Volume Spike
# Uses weekly Williams fractals (bearish for short, bullish for long) as reversal signals,
# weekly EMA20 for trend filter, and volume spike for confirmation.
# Works in bull markets (buy bullish fractal pullbacks in uptrend) and bear markets (sell bearish fractal rallies in downtrend).
# Designed for 15-30 trades/year to avoid fee drag.
name = "6h_WilliamsFractal_WeeklyTrend_Volume"
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
    
    # Get weekly data for Williams fractals and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_6h = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Williams Fractals on weekly data (requires 5 bars: 2 left, center, 2 right)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    # Williams Fractal: bearish = high[n] is highest of [n-2, n-1, n, n+1, n+2]
    # bullish = low[n] is lowest of [n-2, n-1, n, n+1, n+2]
    for i in range(2, len(high_1w) - 2):
        if (high_1w[i] > high_1w[i-2] and high_1w[i] > high_1w[i-1] and 
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]  # bearish fractal at high point
        if (low_1w[i] < low_1w[i-2] and low_1w[i] < low_1w[i-1] and 
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]   # bullish fractal at low point
    
    # Williams fractals need 2 extra weekly bars for confirmation (right side of fractal)
    bearish_fractal_6h = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_6h = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_6h[i]) or np.isnan(bearish_fractal_6h[i]) or 
            np.isnan(bullish_fractal_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Bullish fractal with uptrend and volume spike
            if not np.isnan(bullish_fractal_6h[i]) and close[i] > ema20_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal with downtrend and volume spike
            elif not np.isnan(bearish_fractal_6h[i]) and close[i] < ema20_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below weekly EMA20
            if close[i] < ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above weekly EMA20
            if close[i] > ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals