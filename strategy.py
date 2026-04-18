#!/usr/bin/env python3
"""
6h_Williams_Fractal_MultiTF_Trend
Hypothesis: Uses daily Williams Fractals (confirmed with 2-bar delay) for structure, combined with 6-hour EMA trend filter and volume confirmation.
Enters long when price breaks above bullish fractal resistance with EMA9 > EMA21 and volume spike; short when breaks below bearish fractal support with EMA9 < EMA21 and volume spike.
Fractals provide natural support/resistance levels that work in both trending and ranging markets. Designed for ~15-30 trades/year with low frequency and high win rate.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf, align_htf_to_ltf

def calculate_williams_fractals(high, low, n):
    """Calculate Williams Fractals - bearish (sell) and bullish (buy) fractals."""
    bearish = np.zeros(len(high), dtype=bool)
    bullish = np.zeros(len(low), dtype=bool)
    
    for i in range(2, len(high) - 2):
        # Bearish fractal: high[i] is highest among 2 bars on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = True
            
        # Bullish fractal: low[i] is lowest among 2 bars on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = True
            
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for fractal calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = calculate_williams_fractals(
        df_1d['high'].values, 
        df_1d['low'].values
    )
    
    # Align fractals to 6h timeframe with 2-bar additional delay for confirmation
    bearish_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2
    )
    bullish_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2
    )
    
    # EMA trend filter on 6h data
    ema9 = np.full(n, np.nan)
    ema21 = np.full(n, np.nan)
    k9 = 2 / (9 + 1)
    k21 = 2 / (21 + 1)
    
    for i in range(21, n):
        if i == 21:
            ema9[i] = np.mean(close[i-9+1:i+1])
            ema21[i] = np.mean(close[i-21+1:i+1])
        else:
            if not np.isnan(ema9[i-1]):
                ema9[i] = close[i] * k9 + ema9[i-1] * (1 - k9)
            if not np.isnan(ema21[i-1]):
                ema21[i] = close[i] * k21 + ema21[i-1] * (1 - k21)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Get current fractal levels (only valid when fractal exists)
        bullish_level = bullish_aligned[i] if bullish_aligned[i] > 0 else np.nan
        bearish_level = bearish_aligned[i] if bearish_aligned[i] > 0 else np.nan
        
        if position == 0:
            # Long: break above bullish fractal resistance with uptrend and volume spike
            if (not np.isnan(bullish_level) and 
                close[i] > bullish_level and 
                ema9[i] > ema21[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below bearish fractal support with downtrend and volume spike
            elif (not np.isnan(bearish_level) and 
                  close[i] < bearish_level and 
                  ema9[i] < ema21[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below EMA9 or trend weakens
            if close[i] < ema9[i] or ema9[i] <= ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above EMA9 or trend weakens
            if close[i] > ema9[i] or ema9[i] >= ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_MultiTF_Trend"
timeframe = "6h"
leverage = 1.0