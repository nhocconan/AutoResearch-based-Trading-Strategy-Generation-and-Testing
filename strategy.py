#/usr/bin/env python3
name = "6h_RVI_Divergence_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def relative_vigor_index(close, open, high, low, length=10):
    """Calculate Relative Vigor Index (RVI)"""
    numerator = (close - open) + 2 * (np.roll(close, 1) - np.roll(open, 1)) + \
                2 * (np.roll(close, 2) - np.roll(open, 2)) + (np.roll(close, 3) - np.roll(open, 3))
    denominator = (high - low) + 2 * (np.roll(high, 1) - np.roll(low, 1)) + \
                  2 * (np.roll(high, 2) - np.roll(low, 2)) + (np.roll(high, 3) - np.roll(low, 3))
    
    # Avoid division by zero
    numerator = np.where(denominator == 0, 0, numerator)
    denominator = np.where(denominator == 0, 1, denominator)
    
    rvi_raw = numerator / denominator
    # Smooth with SMA
    rvi = pd.Series(rvi_raw).rolling(window=length, min_periods=length).mean().values
    signal_line = pd.Series(rvi).rolling(window=4, min_periods=4).mean().values
    return rvi, signal_line

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Load 1d data ONCE for trend filter (100 EMA)
    df_1d = get_htf_data(prices, '1d')
    ema100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 2. Calculate RVI on 6h data
    rvi, rvi_signal = relative_vigor_index(close, open_, high, low, length=10)
    
    # 3. Volume filter: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20 * 1.5
    
    # Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30  # Need enough for RVI calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema100_1d_aligned[i]) or np.isnan(rvi[i]) or 
            np.isnan(rvi_signal[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema100_1d_aligned[i]
        price_below_ema1d = close[i] < ema100_1d_aligned[i]
        rvi_bullish = rvi[i] > rvi_signal[i]
        rvi_bearish = rvi[i] < rvi_signal[i]
        
        if position == 0:
            # Long: RVI bullish crossover + above 1d EMA100 + volume spike
            if rvi_bullish and rvi[i-1] <= rvi_signal[i-1] and price_above_ema1d and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: RVI bearish crossover + below 1d EMA100 + volume spike
            elif rvi_bearish and rvi[i-1] >= rvi_signal[i-1] and price_below_ema1d and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: RVI crosses back in opposite direction
            if position == 1:
                if rvi[i] < rvi_signal[i]:  # RVI turns bearish
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if rvi[i] > rvi_signal[i]:  # RVI turns bullish
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals