#!/usr/bin/env python3
# 4h_momentum_follow_volume_v1
# Hypothesis: On 4h timeframe, momentum breakouts (price > highest high of last 20 bars) with volume expansion and 1d EMA50 trend alignment capture trend moves in both bull and bear markets. Trend filter prevents counter-trend entries. Volume filter ensures breakout validity. Designed for low trade frequency and high edge.
# Entry: Long when price > 20-bar high + volume > 1.5x 20-period average + price > 1d EMA50
# Entry: Short when price < 20-bar low + volume > 1.5x 20-period average + price < 1d EMA50
# Exit: Price crosses back below 20-bar high (long) or above 20-bar low (short) or trend reversal
# Position sizing: 0.25 long, -0.25 short

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_momentum_follow_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-bar high and low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < 20-bar high OR price below 1d EMA50
            if (close[i] < high_20[i]) or (close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price > 20-bar low OR price above 1d EMA50
            if (close[i] > low_20[i]) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price > 20-bar high + volume + price > 1d EMA50
            if (close[i] > high_20[i]) and volume_filter[i] and (close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < 20-bar low + volume + price < 1d EMA50
            elif (close[i] < low_20[i]) and volume_filter[i] and (close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals