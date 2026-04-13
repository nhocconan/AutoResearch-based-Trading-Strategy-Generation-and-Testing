#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and volume confirmation
    # Long: price breaks above Donchian upper AND ATR(14) > ATR(50) AND volume > 1.5x avg
    # Short: price breaks below Donchian lower AND ATR(14) > ATR(50) AND volume > 1.5x avg
    # Exit: price reverts to Donchian midpoint OR ATR contraction
    # Using 6h timeframe for moderate trade frequency, Donchian for structure break,
    # ATR regime filter to trade only in expanding volatility, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            highest_high[i] = np.max(high[i-lookback+1:i+1])
            lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate ATR(14) and ATR(50) for volatility regime filter
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.append([close[0]], close[:-1])
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation using Wilder's smoothing
    def atr_calc(data, period):
        result = np.full(n, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, n):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14 = atr_calc(tr, 14)
    atr_50 = atr_calc(tr, 50)
    
    # ATR regime: expanding volatility (short-term > long-term)
    atr_expanding = atr_14 > atr_50
    
    # Get daily data for HTF confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for additional filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    prev_close_1d = np.append([close_1d[0]], close_1d[:-1])
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - prev_close_1d)
    tr3_1d = np.abs(low_1d - prev_close_1d)
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    atr_1d = atr_calc(tr_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 6h volume confirmation (>1.5x 20-period average)
    vol_ma_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_6h[i] = np.mean(volume[i-20:i])
    volume_spike_6h = volume > (1.5 * vol_ma_6h)
    
    # Donchian breakout conditions
    upper_break = close > highest_high
    lower_break = close < lowest_low
    midpoint = (highest_high + lowest_low) / 2
    midpoint_reversion = np.abs(close - midpoint) < (highest_high - lowest_low) * 0.1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_expanding[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(volume_spike_6h[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(upper_break[i]) or np.isnan(lower_break[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + ATR expanding + volume confirmation
        long_entry = upper_break[i] and atr_expanding[i] and volume_spike_6h[i] and volume_spike_1d_aligned[i] > 0.5
        short_entry = lower_break[i] and atr_expanding[i] and volume_spike_6h[i] and volume_spike_1d_aligned[i] > 0.5
        
        # Exit conditions: midpoint reversion OR ATR contraction OR volume dry-up
        long_exit = midpoint_reversion[i] or not atr_expanding[i] or not volume_spike_6h[i]
        short_exit = midpoint_reversion[i] or not atr_expanding[i] or not volume_spike_6h[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_donchian_atr_volume_v1"
timeframe = "6h"
leverage = 1.0