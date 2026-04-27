#!/usr/bin/env python3
"""
12h_Pivot_Squeeze_Reversal
Hypothesis: In low-volatility (squeeze) regimes on 12h, price often reverses sharply
after breaking out of Bollinger Bands. Combines Bollinger Band squeeze detection
with 1d VWAP mean reversion and volume confirmation for high-probability reversals.
Works in both bull and bear markets as it captures mean reversion moves.
Target: 15-25 trades/year per symbol.
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
    
    # Get 1d data for VWAP and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d VWAP
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap = (vwap_numerator / vwap_denominator).values
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width < 20-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Align 1d indicators to 12h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze.astype(float))
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(squeeze_aligned[i])):
            signals[i] = 0.0
            continue
        
        vwap_val = vwap_aligned[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        bb_middle_val = bb_middle_aligned[i]
        squeeze_val = squeeze_aligned[i] > 0.5
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Look for squeeze breakout with volume confirmation
            if squeeze_val and vol_confirm_val:
                # Long: break above upper BB with close > VWAP (bullish rejection)
                if close[i] > bb_upper_val and close[i] > vwap_val:
                    signals[i] = size
                    position = 1
                # Short: break below lower BB with close < VWAP (bearish rejection)
                elif close[i] < bb_lower_val and close[i] < vwap_val:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price returns to VWAP or breaks below middle BB
            if close[i] <= vwap_val or close[i] < bb_middle_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to VWAP or breaks above middle BB
            if close[i] >= vwap_val or close[i] > bb_middle_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Pivot_Squeeze_Reversal"
timeframe = "12h"
leverage = 1.0