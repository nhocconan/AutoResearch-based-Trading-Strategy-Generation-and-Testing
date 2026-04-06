#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA(200) trend filter with volume confirmation
# Enter long when: Williams %R < -80 (oversold), price > 1d EMA(200), volume > 1.5x average
# Enter short when: Williams %R > -20 (overbought), price < 1d EMA(200), volume > 1.5x average
# Exit when: Williams %R returns to -50 (mean reversion) or opposite signal occurs
# Williams %R identifies exhaustion points in trends, effective in both bull and bear markets
# Trend filter prevents counter-trend trades, volume confirms genuine interest
# Target: 50-150 trades over 4 years (12-37/year)

name = "6h_williamsr_1dema200_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1d EMA(200) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Williams %R returns to -50 (mean reversion) or overbought signal
            if williams_r[i] >= -50 or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R returns to -50 (mean reversion) or oversold signal
            if williams_r[i] <= -50 or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extreme + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if williams_r[i] < -80 and close[i] > ema_200_aligned[i]:
                    # Oversold with uptrend - long
                    signals[i] = 0.25
                    position = 1
                elif williams_r[i] > -20 and close[i] < ema_200_aligned[i]:
                    # Overbought with downtrend - short
                    signals[i] = -0.25
                    position = -1
    
    return signals