#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h CAMARILLA R3/L3 breakout with daily volume confirmation and ATR volatility filter.
# CAMARILLA levels provide high-probability support/resistance based on prior day's range.
# Breaking R3 (resistance 3) or L3 (support 3) indicates strong momentum.
# Daily volume filter ensures participation from higher timeframe participants.
# ATR filter avoids choppy markets. Designed for low trade frequency (20-40/year).
# Works in bull markets (breakouts above R3) and bear markets (breakdowns below L3).
name = "4h_Camarilla_R3L3_DailyVolume_ATR_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for CAMARILLA calculation and ATR filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily CAMARILLA levels using previous day's data to avoid look-ahead
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # CAMARILLA calculations
    range_prev = high_prev - low_prev
    # R3 and L3 levels
    r3 = close_prev + range_prev * 1.1 / 2
    l3 = close_prev - range_prev * 1.1 / 2
    
    # Calculate daily ATR (14-period) for volatility filter
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = atr[i-1] * (1 - 1/atr_period) + tr[i] * (1/atr_period)
            else:
                atr[i] = np.nan
    
    # ATR multiplier for volatility filter
    atr_mult = 1.5
    atr_threshold = atr * atr_mult
    
    # Align daily levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1d, atr_threshold)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(atr_threshold_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Volatility filter: ATR threshold must be positive (sufficient volatility)
        vol_filter = not np.isnan(atr_threshold_aligned[i]) and atr_threshold_aligned[i] > 0
        
        if position == 0:
            # Long: price breaks above R3 AND volume confirmation AND volatility filter
            long_breakout = close[i] > r3_aligned[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] < l3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] < l3_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R3 OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] > r3_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals