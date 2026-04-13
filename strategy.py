#!/usr/bin/env python3
"""
1D_Weekly_Momentum_With_Volume_Confirmation
Hypothesis: Buy when weekly momentum (price > weekly SMA40) aligns with daily price > daily SMA20 and volume > 1.5x 20-day average, sell when weekly momentum weakens (price < weekly SMA40) and daily price < daily SMA20 with volume confirmation. Uses daily timeframe with weekly trend filter to capture sustained moves in both bull and bear markets while avoiding whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily SMA20 for trend and entry
    close_s = pd.Series(close)
    sma20_daily = close_s.rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Weekly SMA40 for trend
    weekly_close_s = pd.Series(df_1w['close'])
    sma40_weekly = weekly_close_s.rolling(window=40, min_periods=40).mean().values
    sma40_weekly_aligned = align_htf_to_ltf(prices, df_1w, sma40_weekly)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(sma20_daily[i]) or np.isnan(sma40_weekly_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Long signal: weekly uptrend (price > weekly SMA40) + daily price > daily SMA20 + volume expansion
        long_signal = (close[i] > sma40_weekly_aligned[i] and 
                      close[i] > sma20_daily[i] and 
                      volume_expansion[i])
        
        # Short signal: weekly downtrend (price < weekly SMA40) + daily price < daily SMA20 + volume expansion
        short_signal = (close[i] < sma40_weekly_aligned[i] and 
                       close[i] < sma20_daily[i] and 
                       volume_expansion[i])
        
        # Exit conditions: minimum holding period of 3 days reached or opposite signal
        if position == 1 and (bars_since_entry >= 3 or short_signal):
            position = -1 if short_signal else 0
            signals[i] = -position_size if short_signal else 0.0
            bars_since_entry = 0
        elif position == -1 and (bars_since_entry >= 3 or long_signal):
            position = 1 if long_signal else 0
            signals[i] = position_size if long_signal else 0.0
            bars_since_entry = 0
        elif position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
                bars_since_entry = 0
            elif short_signal:
                position = -1
                signals[i] = -position_size
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1D_Weekly_Momentum_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0