#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
# Williams Fractals identify potential reversal points; breakouts from these levels with volume
# and 1d trend alignment capture strong momentum moves. Works in both bull and bear markets
# by following the 1d trend direction (long in uptrend, short in downtrend).
name = "6h_WilliamsFractal_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Williams Fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals: Bearish (sell) fractal = high[n-2] < high[n-1] > high[n] and low[n-2] > low[n-1] < low[n]
    # Bullish (buy) fractal = high[n-2] > high[n-1] < high[n] and low[n-2] < low[n-1] > low[n]
    # We look for the middle bar (n-1) as the fractal point
    n1d = len(high_1d)
    bearish_fractal = np.zeros(n1d, dtype=bool)
    bullish_fractal = np.zeros(n1d, dtype=bool)
    
    for i in range(2, n1d - 2):
        # Bearish fractal: peak with lower highs on both sides and higher lows
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            low_1d[i-2] > low_1d[i-1] and
            low_1d[i] > low_1d[i-1]):
            bearish_fractal[i-1] = True
        
        # Bullish fractal: trough with higher lows on both sides and lower highs
        if (high_1d[i-2] > high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and
            low_1d[i-2] < low_1d[i-1] and
            low_1d[i] < low_1d[i-1]):
            bullish_fractal[i-1] = True
    
    # Convert to price levels: use the high/low of the fractal bar
    bearish_level = np.full(n1d, np.nan)
    bullish_level = np.full(n1d, np.nan)
    bearish_level[bearish_fractal] = high_1d[bearish_fractal]  # resistance level
    bullish_level[bullish_fractal] = low_1d[bullish_fractal]   # support level
    
    # Need 2 additional bars for fractal confirmation (Williams fractals require 2 bars after)
    bearish_level_confirmed = align_htf_to_ltf(prices, df_1d, bearish_level, additional_delay_bars=2)
    bullish_level_confirmed = align_htf_to_ltf(prices, df_1d, bullish_level, additional_delay_bars=2)
    
    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: volume > 1.8x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(bearish_level_confirmed[i]) or np.isnan(bullish_level_confirmed[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal level (support turned resistance) with volume and above daily EMA50
            if (price > bullish_level_confirmed[i] and vol_spike[i] and price > ema_50_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal level (resistance turned support) with volume and below daily EMA50
            elif (price < bearish_level_confirmed[i] and vol_spike[i] and price < ema_50_6h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below bearish fractal level (support)
            if price < bearish_level_confirmed[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above bullish fractal level (resistance)
            if price > bullish_level_confirmed[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals