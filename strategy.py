#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter
# Long when price breaks above Camarilla R3 (1d) AND 12h volume > 1.5 * avg_volume(20) AND 1d ADX > 25 (trending)
# Short when price breaks below Camarilla S3 (1d) AND 12h volume > 1.5 * avg_volume(20) AND 1d ADX > 25 (trending)
# Exit when price crosses Camarilla H3/L3 (midpoint) OR ADX < 20 (range regime)
# Uses discrete sizing 0.25 to limit fee churn
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla levels provide high-probability reversal/breakout points
# Volume confirmation ensures breakout strength while limiting false signals
# ADX regime filter (25/20 hysteresis) avoids whipsaws in ranging markets
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "12h_Camarilla_R3S3_Breakout_1dADX_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Camarilla and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for Camarilla and ADX
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    # Camarilla R3 = close + 1.1*(high-low)/2
    # Camarilla S3 = close - 1.1*(high-low)/2
    # Camarilla H3 = close + 1.1*(high-low)/4
    # Camarilla L3 = close - 1.1*(high-low)/4
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First bar uses current values
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_R3 = prev_close_1d + 1.1 * camarilla_range / 2
    camarilla_S3 = prev_close_1d - 1.1 * camarilla_range / 2
    camarilla_H3 = prev_close_1d + 1.1 * camarilla_range / 4
    camarilla_L3 = prev_close_1d - 1.1 * camarilla_range / 4
    
    # Align 1d Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(np.diff(low_1d))
    tr3 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(np.diff(close_1d))  # Simplified TR for close-only approximation
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.mean(tr[:3]) if len(tr) >= 3 else tr[0] if len(tr) > 0 else 0], tr])
    
    # Directional Movement
    up_move = np.diff(high_1d)
    down_move = -np.diff(low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    if len(tr) >= 14 and len(plus_dm) >= 14 and len(minus_dm) >= 14:
        atr = wilders_smoothing(tr, 14)
        plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = wilders_smoothing(dx, 14)
    else:
        adx = np.full(len(close_1d), 20.0)  # Default to ranging
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, volume spike, ADX > 25 (trending)
            if (close[i] > camarilla_R3_aligned[i] and 
                volume_confirm[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, volume spike, ADX > 25 (trending)
            elif (close[i] < camarilla_S3_aligned[i] and 
                  volume_confirm[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla H3 OR ADX < 20 (range)
            if close[i] < camarilla_H3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla L3 OR ADX < 20 (range)
            if close[i] > camarilla_L3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals