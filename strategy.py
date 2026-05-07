#!/usr/bin/env python3
# 4H_MultiTimeframe_Confluence_Strategy
# Hypothesis: 4h strategy combining 1d trend filter (EMA34) with 4h breakout signals (Donchian channel breakout) and volume confirmation.
# Goes long when: price breaks above 4h Donchian(20) high, close > 1d EMA34 (uptrend), and volume > 1.5x 20-period average.
# Goes short when: price breaks below 4h Donchian(20) low, close < 1d EMA34 (downtrend), and volume > 1.5x 20-period average.
# Exits when price returns to the opposite Donchian level (middle of channel).
# Designed to capture medium-term trends while avoiding counter-trend trades and minimizing false breakouts.
# Targets 20-40 trades/year on 4h timeframe to stay within optimal range and minimize fee drag.
# Uses 1d EMA for trend filter to work in both bull and bear markets by only trading in direction of higher timeframe trend.

name = "4H_MultiTimeframe_Confluence_Strategy"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA34 for 1d trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 4h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Volume spike detection: 1.5x average volume (20-period for stability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure we have Donchian, EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high, price above 1d EMA34 (uptrend), volume spike
            if (high[i] > donchian_high[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low, price below 1d EMA34 (downtrend), volume spike
            elif (low[i] < donchian_low[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below Donchian middle (mean reversion within channel)
            if close[i] <= donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above Donchian middle
            if close[i] >= donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals