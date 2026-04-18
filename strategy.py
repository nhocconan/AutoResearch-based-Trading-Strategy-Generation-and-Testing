#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with daily EMA trend filter and volume confirmation.
# Camarilla levels (R3/S3) act as mean-reversion zones in range-bound markets.
# Daily EMA50 filter ensures trades align with higher-timeframe trend (long above, short below).
# Volume confirmation adds conviction to reversals.
# Designed for low trade frequency (15-30/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
name = "6h_Camarilla_R3S3_Reversal_DailyEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla and EMA (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC (avoid look-ahead)
    # Camarilla formulas: 
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # H2 = close + 0.75*(high-low), L2 = close - 0.75*(high-low)
    # H1 = close + 0.5*(high-low), L1 = close - 0.5*(high-low)
    # Pivot = (high + low + close) / 3
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    high_d_prev = np.concatenate([[np.nan], high_d[:-1]])
    low_d_prev = np.concatenate([[np.nan], low_d[:-1]])
    close_d_prev = np.concatenate([[np.nan], close_d[:-1]])
    
    # Calculate Camarilla levels
    rng = high_d_prev - low_d_prev
    pivot = (high_d_prev + low_d_prev + close_d_prev) / 3.0
    
    h3 = pivot + 1.125 * rng
    l3 = pivot - 1.125 * rng
    h4 = pivot + 1.5 * rng
    l4 = pivot - 1.5 * rng
    
    # Calculate daily EMA50
    ema_50 = pd.Series(close_d_prev).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Camarilla levels and EMA to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price touches S3 (l3) AND above daily EMA50 AND volume confirmation
            long_setup = (low[i] <= l3_aligned[i]) and (close[i] > ema_50_aligned[i])
            if vol_confirm and long_setup:
                signals[i] = 0.25
                position = 1
            # Short: price touches R3 (h3) AND below daily EMA50 AND volume confirmation
            elif vol_confirm and (high[i] >= h3_aligned[i]) and (close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot OR H3 (profit target) OR breaks below L4 (stop)
            exit_condition = (close[i] >= pivot[i]) or (close[i] >= h3_aligned[i]) or (close[i] < l4_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot OR L3 (profit target) OR breaks above H4 (stop)
            exit_condition = (close[i] <= pivot[i]) or (close[i] <= l3_aligned[i]) or (close[i] > h4_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals