#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel (20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper channel AND 1d ADX > 25 AND volume > 2.0 * avg_volume(20) on 12h
# Short when price breaks below 1d Donchian lower channel AND 1d ADX > 25 AND volume > 2.0 * avg_volume(20) on 12h
# Exit when price crosses the 1d Donchian midpoint (mean of upper and lower channel)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian(20) provides clear breakout levels with proven effectiveness
# 1d ADX > 25 ensures we only trade in trending markets, reducing whipsaws
# High volume threshold (2.0x) controls trade frequency while capturing strong breakouts
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by trading with the 1d trend

name = "12h_1dDonchian20_Breakout_1dADX25_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channel and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need at least 30 completed daily bars for ADX(14) with warmup
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20)
    # Upper channel = highest high over last 20 periods
    # Lower channel = lowest low over last 20 periods
    # Middle channel = (upper + lower) / 2
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    upper_20 = high_max_20
    lower_20 = low_min_20
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Align 1d Donchian levels to 12h timeframe (wait for completed 1d bar)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    
    # Calculate 1d ADX(14) trend filter
    # ADX calculation: +DM, -DM, TR, then smoothed to get +DI, -DI, DX, then ADX
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Add initial zero for index alignment
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(plus_dm) < period:
        adx_1d = np.full(len(close_1d), np.nan)
    else:
        smoothed_plus_dm = wilders_smoothing(plus_dm, period)
        smoothed_minus_dm = wilders_smoothing(minus_dm, period)
        smoothed_tr = wilders_smoothing(tr, period)
        
        # Avoid division by zero
        plus_di = 100 * smoothed_plus_dm / np.where(smoothed_tr == 0, 1, smoothed_tr)
        minus_di = 100 * smoothed_minus_dm / np.where(smoothed_tr == 0, 1, smoothed_tr)
        
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
        adx_1d = wilders_smoothing(dx, period)
    
    # Align 1d ADX to 12h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper channel, ADX > 25, volume spike
            if (close[i] > upper_aligned[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower channel, ADX > 25, volume spike
            elif (close[i] < lower_aligned[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below the 1d Donchian middle channel
            if close[i] < middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above the 1d Donchian middle channel
            if close[i] > middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals