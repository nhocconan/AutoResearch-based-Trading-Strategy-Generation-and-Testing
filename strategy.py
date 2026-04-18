#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot (from 12h) breakout with volume confirmation and ATR filter.
# Camarilla levels from 12h provide strong support/resistance based on prior day's range.
# Breakout above R3 or below S3 with volume and volatility confirmation captures strong moves.
# Designed for 6h timeframe to balance trade frequency and signal quality.
# Works in bull markets (breakouts above R3) and bear markets (breakouts below S3).
name = "6h_Camarilla_R3_S3_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla and ATR (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla levels (based on prior 12h bar's range)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot and levels for each 12h bar
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    r3_12h = close_12h + 1.1 * range_12h / 2
    s3_12h = close_12h - 1.1 * range_12h / 2
    
    # Calculate 12h ATR (14-period) for volatility filter
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    atr_12h = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr_12h[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr_12h[i-1]) and not np.isnan(tr[i]):
                atr_12h[i] = atr_12h[i-1] * (1 - 1/atr_period) + tr[i] * (1/atr_period)
            else:
                atr_12h[i] = np.nan
    
    # Align 12h indicators to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Volatility filter: ATR must be above zero (sufficient volatility)
        vol_filter = not np.isnan(atr_12h_aligned[i]) and atr_12h_aligned[i] > 0
        
        if position == 0:
            # Long: price breaks above R3 AND volume confirmation AND volatility filter
            long_breakout = close[i] > r3_12h_aligned[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] < s3_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S3 OR ATR drops to zero (volatility collapse)
            exit_condition = close[i] < s3_12h_aligned[i] or (np.isnan(atr_12h_aligned[i]) or atr_12h_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R3 OR ATR drops to zero (volatility collapse)
            exit_condition = close[i] > r3_12h_aligned[i] or (np.isnan(atr_12h_aligned[i]) or atr_12h_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals