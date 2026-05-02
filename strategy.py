#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Uses Donchian Channel (20) for breakout detection
# 1d EMA50 filter ensures trades align with higher timeframe trend
# Volume spike (2.0x 24-bar MA) confirms institutional participation
# Designed for 75-200 total trades over 4 years (19-50/year) on 4h timeframe
# Works in bull markets (breakouts with trend) and bear markets (mean reversion at channel extremes)

name = "4h_Donchian_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20) on 4h timeframe
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    dc_upper = high_s.rolling(window=20, min_periods=20).max().values
    dc_lower = low_s.rolling(window=20, min_periods=20).min().values
    dc_middle = (dc_upper + dc_lower) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x 24-period average (24*4h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and volume MA)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above upper band AND price > 1d EMA50 (bullish trend) AND volume spike
            if (close[i] > dc_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below lower band AND price < 1d EMA50 (bearish trend) AND volume spike
            elif (close[i] < dc_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below middle band OR price below 1d EMA50 (trend change)
            if close[i] < dc_middle[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above middle band OR price above 1d EMA50 (trend change)
            if close[i] > dc_middle[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals