#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6f Williams %R with 1d EMA(200) trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions in ranging markets.
# EMA(200) filter ensures trades only in direction of long-term trend.
# Volume confirmation filters low-conviction signals.
# Works in bull/bear: trend filter prevents counter-trend trades, Williams %R captures reversals.
# Target: 75-150 total trades over 4 years (19-38/year) to balance signal quality and fee drag.

name = "6h_williamsr_1dema200_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1d EMA(200) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Williams %R exits oversold OR trend changes
            if williams_r[i] > -20 or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R exits overbought OR trend changes
            if williams_r[i] < -80 or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extreme + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if williams_r[i] < -80 and close[i] > ema_200_aligned[i]:
                    # Oversold with uptrend - long
                    signals[i] = 0.25
                    position = 1
                elif williams_r[i] > -20 and close[i] < ema_200_aligned[i]:
                    # Overbought with downtrend - short
                    signals[i] = -0.25
                    position = -1
    
    return signals