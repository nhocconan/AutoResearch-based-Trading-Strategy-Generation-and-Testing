#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + ADX trend filter
# Long when price breaks above Donchian upper band with volume > 2x median and ADX > 25
# Short when price breaks below Donchian lower band with volume > 2x median and ADX > 25
# Uses discrete position sizing (0.25) to limit trade frequency and avoid fee drag
# Designed to capture trends in both bull and bear markets with volume confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian bands
    dc_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe
    dc_upper_aligned = align_htf_to_ltf(prices, df_1d, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_1d, dc_lower)
    
    # 1d ADX for trend strength (14-period)
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - high_1d[i-1]), 
                   abs(low_1d[i] - low_1d[i-1]))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    # Calculate smoothed +DM, -DM, TR
    smoothed_plus_dm = wilders_smoothing(plus_dm, 14)
    smoothed_minus_dm = wilders_smoothing(minus_dm, 14)
    smoothed_tr = wilders_smoothing(tr, 14)
    
    # Calculate +DI, -DI, DX
    plus_di = np.where(smoothed_tr != 0, smoothed_plus_dm / smoothed_tr * 100, 0)
    minus_di = np.where(smoothed_tr != 0, smoothed_minus_dm / smoothed_tr * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, 
                  abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    
    # ADX is smoothed DX
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(dc_upper_aligned[i]) or np.isnan(dc_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian upper, volume spike, ADX > 25
        if (close[i] > dc_upper_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian lower, volume spike, ADX > 25
        elif (close[i] < dc_lower_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              adx_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: price crosses back inside Donchian channels
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < dc_upper_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > dc_lower_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0