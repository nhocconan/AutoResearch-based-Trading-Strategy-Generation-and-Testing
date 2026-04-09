#!/usr/bin/env python3
# 6h_12h1d_camarilla_pivot_v1
# Hypothesis: 6h strategy using 12h and 1d Camarilla pivot levels with volume confirmation.
# Long: Price breaks above 1d R4 level with volume > 1.8x 20-period average AND 12h close > 12h open (bullish candle).
# Short: Price breaks below 1d S4 level with volume > 1.8x 20-period average AND 12h close < 12h open (bearish candle).
# Exit: Price returns to 1d pivot point (PP) or breaks opposite S4/R4 level.
# Uses multi-timeframe confirmation: 1d for key support/resistance, 12h for trend filter, 6h for execution.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.
# Works in bull/bear: Breakouts with volume work in trending markets; pivot mean reversion works in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h1d_camarilla_pivot_v1"
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
    open_price = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + range_1d * 1.1 / 2.0
    s4 = close_1d - range_1d * 1.1 / 2.0
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 12h data for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # 12h close and open for bullish/bearish candle check
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # Align 12h data to 6h timeframe
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    open_12h_aligned = align_htf_to_ltf(prices, df_12h, open_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(close_12h_aligned[i]) or np.isnan(open_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        # 12h candle direction: bullish if close > open, bearish if close < open
        candle_bullish = close_12h_aligned[i] > open_12h_aligned[i]
        candle_bearish = close_12h_aligned[i] < open_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to daily pivot or breaks below S4
            if close[i] <= pivot_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to daily pivot or breaks above R4
            if close[i] >= pivot_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and 12h trend filter
            bullish_breakout = (close[i] > r4_aligned[i]) and volume_confirmed and candle_bullish
            bearish_breakout = (close[i] < s4_aligned[i]) and volume_confirmed and candle_bearish
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals