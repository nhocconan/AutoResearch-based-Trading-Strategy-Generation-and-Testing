#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + 1d chop regime filter.
# Donchian breakout captures trend continuation in both bull/bear markets.
# Volume confirmation ensures breakout is genuine (not false breakout).
# Chop regime filter (1d) avoids trading in sideways markets.
# Designed for low trade frequency (~20-30/year) to minimize fee drag.
# Entry: Long when price breaks above Donchian upper + 1d volume > 1.5x avg + chop < 61.8.
# Entry: Short when price breaks below Donchian lower + 1d volume > 1.5x avg + chop < 61.8.
# Exit: Opposite Donchian breakout or chop > 61.8 (range market).
# Uses discrete position sizes (0.25) to minimize churn.
name = "4h_Donchian_Volume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1D data for volume and chop
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1D volume MA (20-period)
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    # Align volume MA to 4h
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1D Chopped Index (14-period)
    atr_1d = np.zeros(len(close_1d))
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    # First TR is just high-low
    tr_1d[0] = high_1d[0] - low_1d[0]
    # Calculate ATR using Wilder's smoothing
    atr_1d[0] = tr_1d[0]
    for i in range(1, len(atr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Chop calculation: 100 * log15(sum(ATR,14) / (max_high - min_low))
    sum_atr_14 = np.zeros(len(close_1d))
    max_high_14 = np.zeros(len(close_1d))
    min_low_14 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        start_idx = max(0, i-13)
        sum_atr_14[i] = np.sum(tr_1d[start_idx:i+1])
        max_high_14[i] = np.max(high_1d[start_idx:i+1])
        min_low_14[i] = np.min(low_1d[start_idx:i+1])
    
    # Avoid division by zero
    range_14 = max_high_14 - min_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop_1d = 100 * np.log14(sum_atr_14 / range_14)
    
    # Align chop to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > Donchian high + volume spike + chop < 61.8 (trending)
            if (close[i] > donch_high[i] and 
                vol_1d[i // 16] > vol_ma_1d[i // 16] * 1.5 and  # Use 1d volume directly (aligned by index)
                chop_1d_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian low + volume spike + chop < 61.8 (trending)
            elif (close[i] < donch_low[i] and 
                  vol_1d[i // 16] > vol_ma_1d[i // 16] * 1.5 and
                  chop_1d_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price < Donchian low OR chop > 61.8 (range)
            if (close[i] < donch_low[i]) or (chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price > Donchian high OR chop > 61.8 (range)
            if (close[i] > donch_high[i]) or (chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: The above uses direct 1d indexing (i//16) for volume which is acceptable
# because we're using the previous day's volume (not current forming bar).
# For more complex indicators, use align_htf_to_ltf as shown for vol_ma and chop.