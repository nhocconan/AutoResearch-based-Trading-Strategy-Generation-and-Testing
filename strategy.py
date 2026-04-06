#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h trend filter and volume confirmation
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA(20) AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power > 0 AND price < 12h EMA(20) AND volume > 1.5x average
# Exit when Bull Power and Bear Power have same sign (both positive or both negative)
# Uses Elder Ray to measure bull/bear power relative to EMA, with 12h trend filter for higher timeframe bias
# Target: 75-150 total trades over 4 years (19-38/year) for optimal 6h performance

name = "6h_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray Index (13-period EMA as base)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12-hour EMA(20) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema20_12h = close_12h_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema20_12h_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Bull and Bear power have same sign (both bullish or both bearish)
        if position == 1:  # long position
            if bull_power[i] > 0 and bear_power[i] > 0:  # both bullish - overextended
                signals[i] = 0.0
                position = 0
            elif bull_power[i] < 0 and bear_power[i] < 0:  # both bearish - reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bull_power[i] > 0 and bear_power[i] > 0:  # both bullish - reversal
                signals[i] = 0.0
                position = 0
            elif bull_power[i] < 0 and bear_power[i] < 0:  # both bearish - overextended
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Bull Power > 0 (bulls in control) AND Bear Power < 0 (bears weak) 
            #       AND price > 12h EMA (uptrend) AND volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema20_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) AND Bull Power > 0 (bulls weak)
            #        AND price < 12h EMA (downtrend) AND volume confirmation
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  close[i] < ema20_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals