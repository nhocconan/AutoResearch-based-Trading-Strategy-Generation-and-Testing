#!/usr/bin/env python3
# 4h_VolumeSpike_Reversal_1dTrend
# Strategy: Mean reversion on volume spikes with 1d trend filter
# Long when: volume > 2x 20-period average AND price < BB lower AND price > 1d EMA50
# Short when: volume > 2x 20-period average AND price > BB upper AND price < 1d EMA50
# Exit when price returns to middle Bollinger Band
# Uses volume spikes to catch exhaustion moves and trend filter to avoid counter-trend trades
# Designed for 4h timeframe with selective entries to minimize trade frequency

name = "4h_VolumeSpike_Reversal_1dTrend"
timeframe = "4h"
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
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Bollinger Bands (20, 2.0)
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = (sma_20 + 2 * std_20).values
    bb_lower = (sma_20 - 2 * std_20).values
    bb_middle = sma_20.values
    
    # Calculate volume spike (current volume > 2x 20-period average)
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (2 * vol_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: volume spike + price below BB lower + above 1d EMA50 (uptrend filter)
            if volume_spike[i] and close[i] < bb_lower[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: volume spike + price above BB upper + below 1d EMA50 (downtrend filter)
            elif volume_spike[i] and close[i] > bb_upper[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle Bollinger Band
            if close[i] >= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle Bollinger Band
            if close[i] <= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals