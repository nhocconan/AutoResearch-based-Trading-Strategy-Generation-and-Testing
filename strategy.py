#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Bollinger Band squeeze breakout with volume confirmation
# Long when price breaks above upper Bollinger Band (20,2) after low volatility squeeze, with volume > 1.5x average
# Short when price breaks below lower Bollinger Band after squeeze, with volume confirmation
# Bollinger Band squeeze identifies low volatility periods that often precede explosive moves
# Volume confirmation ensures breakouts have institutional participation
# Designed to work in both bull and bear markets by capturing volatility expansion moves
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "12h_1dBB_Squeeze_Breakout_Volume_v1"
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
    
    # Calculate 1-day Bollinger Bands (20-period, 2 std dev)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period SMA and standard deviation for Bollinger Bands
    sma_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    
    # Upper and lower Bollinger Bands
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Bollinger Band Width for squeeze detection (normalized by SMA)
    bb_width = (bb_upper - bb_lower) / sma_20
    
    # Squeeze condition: BB width below 20-period average width (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Align Bollinger Bands and squeeze condition to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Bollinger Band warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper BB after squeeze with volume confirmation
            if close[i] > bb_upper_aligned[i] and squeeze_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower BB after squeeze with volume confirmation
            elif close[i] < bb_lower_aligned[i] and squeeze_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Bollinger Band (mean reversion)
            if close[i] < bb_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Bollinger Band (mean reversion)
            if close[i] > bb_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals