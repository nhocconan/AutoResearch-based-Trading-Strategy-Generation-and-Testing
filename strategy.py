#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w ATR-based volatility filter and 1d Donchian channel breakout.
# Long: Close breaks above 20-day Donchian high + weekly ATR < 0.8 * 50-week ATR (low volatility regime).
# Short: Close breaks below 20-day Donchian low + weekly ATR < 0.8 * 50-week ATR.
# Uses volatility regime filter to avoid false breakouts in high volatility periods.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-day Donchian channel (using previous day's data)
    donch_high = np.full(len(close_1d), np.nan)
    donch_low = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        donch_high[i] = np.max(high_1d[i-20:i])
        donch_low[i] = np.min(low_1d[i-20:i])
    
    # 1w data for ATR-based volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for ATR
    tr_1w = np.zeros(len(close_1w))
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(close_1w)):
        tr_1w[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    # 50-week ATR (slow ATR for regime)
    atr_50w = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        atr_50w[i] = np.mean(tr_1w[i-50:i])
    
    # Current week ATR (fast ATR)
    atr_current = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):  # 2-week ATR for current volatility
        atr_current[i] = np.mean(tr_1w[i-14:i])
    
    # Volatility filter: current ATR < 0.8 * long ATR (low volatility regime)
    vol_filter = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        if not np.isnan(atr_50w[i]) and atr_50w[i] > 0:
            vol_filter[i] = atr_current[i] < 0.8 * atr_50w[i]
        else:
            vol_filter[i] = False
    
    # Align 1d Donchian levels to 1d (no alignment needed for same timeframe)
    donch_high_aligned = donch_high
    donch_low_aligned = donch_low
    
    # Align 1w volatility filter to 1d
    vol_filter_aligned = align_htf_to_ltf(prices, df_1w, vol_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_filt = bool(vol_filter_aligned[i])
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        
        if position == 0:
            # Long: price closes above Donchian high + low volatility regime
            if price > upper and vol_filt:
                position = 1
                signals[i] = position_size
            # Short: price closes below Donchian low + low volatility regime
            elif price < lower and vol_filt:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Donchian low (opposite bound)
            if price < lower:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above Donchian high (opposite bound)
            if price > upper:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Volatility_Filter"
timeframe = "1d"
leverage = 1.0