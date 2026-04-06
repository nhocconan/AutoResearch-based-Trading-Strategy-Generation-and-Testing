#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 12h volume confirmation
# Long when price breaks above 20-period 6h high AND 12h volume > 1.5x 20-period average
# Short when price breaks below 20-period 6h low AND 12h volume > 1.5x 20-period average
# Uses volume filter to avoid false breakouts. Donchian provides clear breakout signals.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_donchian20_12h_vol_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # 12h volume average for confirmation (load once)
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if 12h volume MA data not available
        if np.isnan(volume_ma_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or stop loss (3*ATR approximation)
        if position == 1:  # long position
            # Exit: price breaks below 20-period low
            if close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 20-period high
            if close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: price breaks above 20-period high AND 12h volume > 1.5x average
            if (close[i] > donchian_high[i] and 
                volume_12h[i] > 1.5 * volume_ma_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low AND 12h volume > 1.5x average
            elif (close[i] < donchian_low[i] and 
                  volume_12h[i] > 1.5 * volume_ma_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals