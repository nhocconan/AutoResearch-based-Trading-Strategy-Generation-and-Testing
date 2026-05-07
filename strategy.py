#/usr/bin/env python3
"""
12h_Price_Channel_1dTrend_Volume_Signal
Hypothesis: Use Donchian channel breakout with 1-day trend filter and volume confirmation on 12h timeframe.
Designed to work in both bull and bear markets by filtering with higher timeframe trend.
Target 20-40 trades/year to minimize fee drag.
"""

name = "12h_Price_Channel_1dTrend_Volume_Signal"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel parameters
    donchian_period = 20
    
    # Calculate Donchian upper and lower bands using previous periods
    high_roll = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # 1-day trend filter: EMA of daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donchian_period, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above upper Donchian band AND above 1-day EMA with volume confirmation
            if close[i] > high_roll[i] and close[i] > ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below lower Donchian band AND below 1-day EMA with volume confirmation
            elif close[i] < low_roll[i] and close[i] < ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below lower Donchian band (mean reversion)
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above upper Donchian band (mean reversion)
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals