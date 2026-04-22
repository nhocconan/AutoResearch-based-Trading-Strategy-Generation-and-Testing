#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R with 1-week trend filter and volume confirmation.
Long when Williams %R crosses above -50 (bullish momentum), weekly close > weekly EMA50 (bullish trend), and volume > 1.5x average volume.
Short when Williams %R crosses below -50 (bearish momentum), weekly close < weekly EMA50 (bearish trend), and volume > 1.5x average volume.
Exit when Williams %R crosses back below -20 (for long) or above -80 (for short) or trend reverses.
Williams %R captures momentum extremes; weekly EMA50 filters higher timeframe trend; volume confirmation ensures conviction.
Designed for low trade frequency by requiring multiple confirmations and using extreme levels.
Works in both bull and bear markets by following weekly trend while using 12h Williams %R for entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams %R (14 periods)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Average volume (50 periods)
    avg_volume = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for Williams %R and volume average
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -50, weekly close > weekly EMA50, volume > 1.5x average
            if (williams_r[i] > -50 and williams_r[i-1] <= -50 and  # Cross above -50
                close_1w[i//12] > ema50_1w[i//12] and  # Weekly close > EMA50 (using integer division for alignment)
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -50, weekly close < weekly EMA50, volume > 1.5x average
            elif (williams_r[i] < -50 and williams_r[i-1] >= -50 and  # Cross below -50
                  close_1w[i//12] < ema50_1w[i//12] and  # Weekly close < EMA50
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R falls below -20 OR weekly close < weekly EMA50
                if (williams_r[i] < -20 or 
                    close_1w[i//12] < ema50_1w[i//12]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R rises above -80 OR weekly close > weekly EMA50
                if (williams_r[i] > -80 or 
                    close_1w[i//12] > ema50_1w[i//12]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0