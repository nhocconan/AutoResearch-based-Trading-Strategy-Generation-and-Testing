#!/usr/bin/env python3
"""
4h_MACD_TrendFilter_12hVolumeSpike
Hypothesis: MACD crossover aligned with 12h trend (EMA100) and confirmed by 12h volume spikes captures strong trend continuations with low frequency. Volume spikes filter out false breakouts, while the 12h EMA100 trend filter ensures trades align with the dominant trend. Targets 15-25 trades/year on 4h timeframe.
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
    
    # Get 12h data for trend filter and volume spike
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    # Calculate 12h EMA100 for trend filter
    close_12h = df_12h['close'].values
    ema_100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_100_12h)
    
    # Calculate 12h volume MA20 for volume spike detection
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate MACD (12,26,9)
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - macd_signal
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_100_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(macd_line[i]) or
            np.isnan(macd_signal[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA100
        uptrend = close[i] > ema_100_12h_aligned[i]
        downtrend = close[i] < ema_100_12h_aligned[i]
        
        # MACD crossover
        macd_bullish = macd_line[i] > macd_signal[i] and macd_line[i-1] <= macd_signal[i-1]
        macd_bearish = macd_line[i] < macd_signal[i] and macd_line[i-1] >= macd_signal[i-1]
        
        # Volume confirmation: >2.0x 20-period MA on 12h (significant spike)
        vol_spike = volume[i] > (2.0 * vol_ma_20_12h_aligned[i])
        
        # Entry logic: MACD crossover in direction of trend with volume spike
        long_entry = vol_spike and uptrend and macd_bullish
        short_entry = vol_spike and downtrend and macd_bearish
        
        # Exit logic: opposite MACD crossover
        long_exit = macd_bearish
        short_exit = macd_bullish
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_MACD_TrendFilter_12hVolumeSpike"
timeframe = "4h"
leverage = 1.0