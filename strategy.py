#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band (20-period), price > 1d EMA50, and volume > 1.5x 20-period average
# Short when price breaks below 4h Donchian lower band (20-period), price < 1d EMA50, and volume > 1.5x 20-period average
# Exit when price crosses back below/above 4h Donchian middle band or 1d EMA direction contradicts position
# Position size: 0.25 (25% of capital) to balance return and drawdown
# Designed to work in trending markets via Donchian breakout and 1d EMA filter, with volume confirmation to avoid false breakouts
# Target: 20-50 trades/year per symbol to minimize fee drag

name = "4h_Donchian_1dEMA_Volume_Filter"
timeframe = "4h"
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
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper = high_roll.values
    lower = low_roll.values
    middle = ((upper + lower) / 2)  # Donchian middle band
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, above 1d EMA50, and volume spike
            if (close[i] > upper[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, below 1d EMA50, and volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle band OR 1d EMA turns bearish
            if (close[i] < middle[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle band OR 1d EMA turns bullish
            if (close[i] > middle[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals