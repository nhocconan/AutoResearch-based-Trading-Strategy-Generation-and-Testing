#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_fair_value_gap_breakout
# Uses weekly Fair Value Gap (FVG) levels as dynamic support/resistance.
# Long when price breaks above FVG upper boundary with volume confirmation.
# Short when price breaks below FVG lower boundary with volume confirmation.
# Exits when price returns to opposite FVG boundary (mean reversion).
# Designed for very low trade frequency (target: 7-25 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and ranging markets via mean reversion to FVG.
# FVG identifies institutional order flow imbalances, effective in both bull and bear markets.

name = "1d_1w_fair_value_gap_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for FVG calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 3:
        return np.zeros(n)
    
    # Calculate weekly Fair Value Gaps
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Bullish FVG: low of current candle > high of previous candle
    bullish_fvg_high = high_1w[2:]  # current candle high
    bullish_fvg_low = high_1w[1:-1]  # previous candle high (gap lower bound)
    bullish_fvg = bullish_fvg_high > bullish_fvg_low
    
    # Bearish FVG: high of current candle < low of previous candle
    bearish_fvg_low = low_1w[2:]  # current candle low
    bearish_fvg_high = low_1w[1:-1]  # previous candle low (gap upper bound)
    bearish_fvg = bearish_fvg_low < bearish_fvg_high
    
    # Create arrays for FVG boundaries
    fvg_upper = np.full(len(high_1w), np.nan)
    fvg_lower = np.full(len(high_1w), np.nan)
    
    # Bullish FVG: gap between previous high and current low
    fvg_upper[2:] = np.where(bullish_fvg, high_1w[2:], np.nan)  # upper boundary = current high
    fvg_lower[2:] = np.where(bullish_fvg, high_1w[1:-1], np.nan)  # lower boundary = previous high
    
    # Bearish FVG: gap between previous low and current high
    fvg_upper[2:] = np.where(bearish_fvg & ~np.isnan(fvg_upper[2:]), fvg_upper[2:], low_1w[1:-1])  # upper boundary = previous low
    fvg_lower[2:] = np.where(bearish_fvg & ~np.isnan(fvg_lower[2:]), fvg_lower[2:], low_1w[2:])  # lower boundary = current low
    
    # Align weekly FVG levels to daily timeframe
    fvg_upper_aligned = align_htf_to_ltf(prices, df_1w, fvg_upper)
    fvg_lower_aligned = align_htf_to_ltf(prices, df_1w, fvg_lower)
    
    # Volume confirmation: volume > 2.0 * 50-period average (daily timeframe)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(fvg_upper_aligned[i]) or np.isnan(fvg_lower_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above FVG upper boundary
        if close[i] > fvg_upper_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below FVG lower boundary
        elif close[i] < fvg_lower_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to opposite FVG boundary (mean reversion)
        elif position == 1 and close[i] <= fvg_lower_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= fvg_upper_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals