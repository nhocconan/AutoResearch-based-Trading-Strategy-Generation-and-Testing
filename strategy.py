#!/usr/bin/env python3
# Hypothesis: 12h momentum with 1d trend filter and volume confirmation
# Long when price > 12h EMA20, above 1d EMA50 (trend filter), and volume > 1.8x 20-period average
# Short when price < 12h EMA20, below 1d EMA50, and volume > 1.8x 20-period average
# Exit when price crosses back below/above EMA20 or trend filter fails
# Position size: 0.25 (25% of capital) to limit drawdown and reduce trade frequency
# Designed for trending markets with volume confirmation to avoid false signals

name = "12h_EMA20_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA20 for entry signal
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d EMA50 for trend filter (only use after daily close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above EMA20, above 1d EMA50 (bullish trend), volume spike
            if (close[i] > ema20[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA20, below 1d EMA50 (bearish trend), volume spike
            elif (close[i] < ema20[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA20 OR trend turns bearish
            if (close[i] < ema20[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA20 OR trend turns bullish
            if (close[i] > ema20[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals