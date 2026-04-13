# 1d_1w_CamarillaBreakout_TrendFilter
# Breakout of daily Camarilla H3/L3 levels with volume spike, filtered by weekly trend (price above/below weekly 200 EMA).
# Targets 8-20 trades/year (30-80 total over 4 years) on 1d timeframe.
# Uses weekly trend filter to avoid counter-trend trades in strong trends, improving win rate in both bull and bear markets.

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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for previous day (H3/L3)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    h3 = pivot + (range_val * 1.1 / 4)
    l3 = pivot - (range_val * 1.1 / 4)
    
    # Breakout conditions
    breakout_up = high_1d > h3
    breakout_down = low_1d < l3
    
    # Align breakout signals to 1d timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up.astype(float))
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down.astype(float))
    
    # Volume spike: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly 200 EMA for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Trend: price above EMA = uptrend, below EMA = downtrend
    uptrend_1w = close_1w > ema_200_1w
    downtrend_1w = close_1w < ema_200_1w
    
    # Align weekly trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla breakout + volume spike + weekly trend alignment
        long_entry = (breakout_up_aligned[i] > 0.5 and 
                      vol_spike_aligned[i] > 0.5 and 
                      uptrend_1w_aligned[i] > 0.5)
        short_entry = (breakout_down_aligned[i] > 0.5 and 
                       vol_spike_aligned[i] > 0.5 and 
                       downtrend_1w_aligned[i] > 0.5)
        
        # Exit when price returns to daily pivot point
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        
        exit_long = position == 1 and close[i] <= pivot_aligned[i]
        exit_short = position == -1 and close[i] >= pivot_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
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

name = "1d_1w_CamarillaBreakout_TrendFilter"
timeframe = "1d"
leverage = 1.0