# 4h_Camarilla_R1S1_Breakout_Volume_1d_Trend_Filter
# Hypothesis: Trade breakouts at Camarilla R1/S1 levels on 4h with volume confirmation and 1d trend filter (EMA50).
# In bull markets: long R1 breakouts with volume and 1d uptrend; short S1 breakdowns with volume and 1d downtrend.
# In bear markets: same logic applies - trend filter prevents counter-trend trades.
# Volume confirmation reduces false breakouts. Target 20-40 trades/year per symbol.
# Uses 1d EMA50 for trend filter to avoid overtrading vs 12h version.

name = "4h_Camarilla_R1S1_Breakout_Volume_1d_Trend_Filter"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume spike (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    # Calculate Camarilla levels from previous day using actual 1d data
    # Shift 1d data to get previous day's OHLC
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    # First element stays as first day's value (no look-ahead)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    close_1d_shift[0] = close_1d[0]
    
    # Previous day's range
    range_1d = high_1d_shift - low_1d_shift
    
    # Align 1d data to 4h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d_shift)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d_shift)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_shift)
    range_1d_aligned = align_htf_to_ltf(prices, df_1d, range_1d)
    
    # Camarilla R1 and S1 levels
    R1 = close_1d_aligned + 1.1 * range_1d_aligned / 12
    S1 = close_1d_aligned - 1.1 * range_1d_aligned / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND 1d uptrend (price > EMA50)
            if close[i] > R1[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND 1d downtrend (price < EMA50)
            elif close[i] < S1[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR 1d trend turns down
            if close[i] < S1[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR 1d trend turns up
            if close[i] > R1[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals