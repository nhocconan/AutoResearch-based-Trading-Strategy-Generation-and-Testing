# 4h_1d_camarilla_breakout_v1
# Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
# In ranging markets, price reverts to mean (H4/H3 or L3/L4). In trending markets,
# breakouts beyond H5/L5 with volume confirmation capture momentum.
# Works in both bull/bear: mean reversion in range, breakout in trend.
# Uses volume filter to reduce false signals and control trade frequency.

#!/usr/bin/env python3
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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, H5, L3, L4, L5
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_h5 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    camarilla_l5 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        range_ = high_1d[i-1] - low_1d[i-1]
        close_prev = close_1d[i-1]
        
        camarilla_h4[i] = close_prev + range_ * 1.1 / 2
        camarilla_h3[i] = close_prev + range_ * 1.1 / 4
        camarilla_h5[i] = close_prev + range_ * 1.1 * 2
        camarilla_l3[i] = close_prev - range_ * 1.1 / 4
        camarilla_l4[i] = close_prev - range_ * 1.1 / 2
        camarilla_l5[i] = close_prev - range_ * 1.1 * 2
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Volume average for confirmation (20-period)
    vol_ma = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ma_values = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ma[:] = vol_ma_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average
        vol_ok = volume[i] > vol_ma[i]
        
        # Mean reversion conditions (range-bound)
        long_mr = (close[i] <= l4_aligned[i] and close[i] > l3_aligned[i]) and vol_ok
        short_mr = (close[i] >= h4_aligned[i] and close[i] < h3_aligned[i]) and vol_ok
        
        # Breakout conditions (trending)
        long_breakout = (close[i] > h5_aligned[i]) and vol_ok
        short_breakout = (close[i] < l5_aligned[i]) and vol_ok
        
        # Exit conditions
        exit_long = position == 1 and (close[i] >= h3_aligned[i] or close[i] <= l3_aligned[i])
        exit_short = position == -1 and (close[i] <= l3_aligned[i] or close[i] >= h3_aligned[i])
        
        # Execute signals
        if long_mr and position != 1:
            position = 1
            signals[i] = position_size
        elif short_mr and position != -1:
            position = -1
            signals[i] = -position_size
        elif long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0