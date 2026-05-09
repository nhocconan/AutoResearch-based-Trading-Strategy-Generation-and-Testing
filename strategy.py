#!/usr/bin/env python3
# Hypothesis: 12-hour timeframe strategy using 1-day Williams %R for overbought/oversold conditions
# combined with 4-hour RSI momentum confirmation and volume filter.
# In ranging markets (Williams %R between -80 and -20), we take mean-reversion trades:
# - Long when Williams %R < -50 (oversold) AND 4h RSI < 40 (momentum exhaustion) AND volume > 1.5x average
# - Short when Williams %R > -50 (overbought) AND 4h RSI > 60 (momentum exhaustion) AND volume > 1.5x average
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
# Uses 1-day Williams %R for higher timeframe context and 4-hour RSI for entry timing.
# Volume filter ensures trades occur during active market participation.
# Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.

name = "12h_WilliamsR_RSI_Volume_MeanReversion"
timeframe = "12h"
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
    
    # Calculate 1-day Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = high_1d.rolling(window=14, min_periods=14).max()
    lowest_low = low_1d.rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r_values = williams_r.values
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_values)
    
    # Calculate 4-hour RSI (14-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    close_4h = df_4h['close']
    delta = close_4h.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_values)
    
    # Calculate volume moving average (20-period) for volume filter
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R oversold (< -50) AND RSI weak (< 40) AND volume above average
            if williams_r_aligned[i] < -50 and rsi_aligned[i] < 40 and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -50) AND RSI strong (> 60) AND volume above average
            elif williams_r_aligned[i] > -50 and rsi_aligned[i] > 60 and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (overbought territory)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (oversold territory)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals