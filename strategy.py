#165164
# Strategy: 1d_Pivot_Reversal_With_Volume_Spike
# Hypothesis: Daily Camarilla pivot levels (R3/S3) act as strong support/resistance.
# A reversal signal occurs when price touches R3/S3 with volume confirmation (>2x average).
# Uses weekly trend filter (price above/below weekly EMA50) to align with higher timeframe trend.
# Designed for low trade frequency (<25/year) to minimize fee drag on daily timeframe.
# Works in both bull and bear markets by trading reversals at extreme levels.

name = "1d_Pivot_Reversal_With_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    pivot = (high + low + close) / 3
    range_val = high - low
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    return R3, S3  # Using R3/S3 as primary reversal levels

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily Camarilla levels (R3, S3) - using previous day's HLC
    # Shift by 1 to avoid look-ahead (use previous day's data)
    R3, S3 = calculate_camarilla(high[:-1], low[:-1], close[:-1])
    # Prepend first value to maintain array length (no trade on first bar)
    R3 = np.concatenate([[np.nan], R3])
    S3 = np.concatenate([[np.nan], S3])
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1st index (0 has no prior day)
        if np.isnan(R3[i]) or np.isnan(S3[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Price touches or goes below S3 (support) with volume confirmation
            # AND weekly trend is bullish (price above weekly EMA50)
            if (low[i] <= S3[i] and 
                volume_filter[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or goes above R3 (resistance) with volume confirmation
            # AND weekly trend is bearish (price below weekly EMA50)
            elif (high[i] >= R3[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily pivot (mean reversion)
            # OR weekly trend turns bearish
            daily_pivot = (high[i-1] + low[i-1] + close[i-1]) / 3
            if (close[i] >= daily_pivot) or (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to daily pivot
            # OR weekly trend turns bullish
            daily_pivot = (high[i-1] + low[i-1] + close[i-1]) / 3
            if (close[i] <= daily_pivot) or (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
#165164
# Strategy: 1d_Pivot_Reversal_With_Volume_Spike
# Hypothesis: Daily Camarilla pivot levels (R3/S3) act as strong support/resistance.
# A reversal signal occurs when price touches R3/S3 with volume confirmation (>2x average).
# Uses weekly trend filter (price above/below weekly EMA50) to align with higher timeframe trend.
# Designed for low trade frequency (<25/year) to minimize fee drag on daily timeframe.
# Works in both bull and bear markets by trading reversals at extreme levels.

name = "1d_Pivot_Reversal_With_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    pivot = (high + low + close) / 3
    range_val = high - low
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    return R3, S3  # Using R3/S3 as primary reversal levels

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily Camarilla levels (R3, S3) - using previous day's HLC
    # Shift by 1 to avoid look-ahead (use previous day's data)
    R3, S3 = calculate_camarilla(high[:-1], low[:-1], close[:-1])
    # Prepend first value to maintain array length (no trade on first bar)
    R3 = np.concatenate([[np.nan], R3])
    S3 = np.concatenate([[np.nan], S3])
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1st index (0 has no prior day)
        if np.isnan(R3[i]) or np.isnan(S3[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Price touches or goes below S3 (support) with volume confirmation
            # AND weekly trend is bullish (price above weekly EMA50)
            if (low[i] <= S3[i] and 
                volume_filter[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or goes above R3 (resistance) with volume confirmation
            # AND weekly trend is bearish (price below weekly EMA50)
            elif (high[i] >= R3[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily pivot (mean reversion)
            # OR weekly trend turns bearish
            daily_pivot = (high[i-1] + low[i-1] + close[i-1]) / 3
            if (close[i] >= daily_pivot) or (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to daily pivot
            # OR weekly trend turns bullish
            daily_pivot = (high[i-1] + low[i-1] + close[i-1]) / 3
            if (close[i] <= daily_pivot) or (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals