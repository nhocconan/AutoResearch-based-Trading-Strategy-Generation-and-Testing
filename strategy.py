#!/usr/bin/env python3
"""
Hypothesis: In BTC/ETH markets, price tends to respect the 200-day EMA as a dynamic trend filter.
This strategy combines daily pivot levels with 200-day EMA trend filter and volume confirmation
to capture high-probability breakouts in the direction of the long-term trend.
- Long when: price breaks above daily pivot + volume > 2x average + price above 200-day EMA
- Short when: price breaks below daily pivot + volume > 2x average + price below 200-day EMA
- Exit when: price returns to the midpoint between pivot and prior day's opposite extreme
Designed for 4h timeframe to work in both bull (trend-following) and bear (counter-trend to 200EMA) regimes.
Target: 20-30 trades per year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and 200-day EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot and support/resistance levels
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    
    # Define exit levels: midpoint between pivot and prior day's high/low
    upper_exit = (pivot + phigh) / 2
    lower_exit = (pivot + plow) / 2
    
    # Calculate 200-day EMA for trend filter
    ema_200 = pd.Series(pclose).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all daily levels to 4h timeframe (waits for daily bar to close)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    upper_exit_4h = align_htf_to_ltf(prices, df_1d, upper_exit)
    lower_exit_4h = align_htf_to_ltf(prices, df_1d, lower_exit)
    ema_200_4h = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for 200-day EMA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_4h[i]) or np.isnan(upper_exit_4h[i]) or np.isnan(lower_exit_4h[i]) or
            np.isnan(ema_200_4h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above pivot with volume spike and above 200-day EMA
            if price > pivot_4h[i] and vol > 2.0 * vol_ma and price > ema_200_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below pivot with volume spike and below 200-day EMA
            elif price < pivot_4h[i] and vol > 2.0 * vol_ma and price < ema_200_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to upper exit level (midpoint between pivot and prior high)
            if price < upper_exit_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to lower exit level (midpoint between pivot and prior low)
            if price > lower_exit_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_200EMA_Volume_MidExit"
timeframe = "4h"
leverage = 1.0