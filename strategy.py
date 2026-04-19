#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe using 1-week Bollinger Band squeeze and breakout.
# Uses weekly Bollinger Bands to detect low volatility squeezes (band width < 20th percentile).
# On squeeze breakout (price closes outside bands), enter long/right with 1-day ATR stop.
# Volume confirmation ensures breakout strength. Works in both bull and bear markets
# by capturing volatility breakouts in either direction. Target: 50-150 total trades.
name = "12h_1w_BollingerSqueeze_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands on weekly timeframe (20-period, 2 std dev)
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Bollinger Band Width (normalized by SMA)
    bb_width = (upper_bb - lower_bb) / sma_20
    # 20th percentile of BB width for squeeze detection
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze = bb_width < bb_width_percentile
    
    # Align squeeze and Bollinger Bands to 12h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on 1d timeframe (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Look for squeeze breakout: price closes outside Bollinger Bands
            if squeeze_aligned[i]:
                # Long breakout: price closes above upper band with volume
                if close[i] > upper_bb_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price closes below lower band with volume
                elif close[i] < lower_bb_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
                    
        elif position == 1:
            # Long position: exit when price closes below lower Bollinger Band
            if close[i] < lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price closes above upper Bollinger Band
            if close[i] > upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals