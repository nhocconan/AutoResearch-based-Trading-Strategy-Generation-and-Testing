#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h Williams %R + 1d EMA(200) + volume confirmation.
# Williams %R identifies oversold/overbought conditions in the 12h trend.
# EMA(200) on 1d provides long-term trend filter to avoid counter-trend trades.
# Volume > 1.5x average confirms momentum behind the move.
# Works in bull via mean reversion in uptrend, in bear via mean reversion in downtrend.
# Target: 80-160 total trades over 4 years (20-40/year) to balance signal quality and fee drag.

name = "6h_williamsr_1dema200_vol_v1"
timeframe = "6h"
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
    
    # 12h Williams %R (14-period) for mean reversion signals
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # 1d EMA(200) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Williams %R crosses above -20 (overbought) OR price below EMA200
            if williams_r_aligned[i] >= -20 or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses below -80 (oversold) OR price above EMA200
            if williams_r_aligned[i] <= -80 or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extreme + EMA200 trend + volume
            if volume[i] > volume_threshold[i]:
                if williams_r_aligned[i] <= -80 and close[i] > ema_200_aligned[i]:
                    # Oversold in 12h with 1d uptrend: long mean reversion
                    signals[i] = 0.25
                    position = 1
                elif williams_r_aligned[i] >= -20 and close[i] < ema_200_aligned[i]:
                    # Overbought in 12h with 1d downtrend: short mean reversion
                    signals[i] = -0.25
                    position = -1
    
    return signals