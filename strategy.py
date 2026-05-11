#!/usr/bin/env python3
name = "1d_WeeklyDonchianBreakout_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly data for Donchian and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian Channel (20-period)
    upper_20 = np.full_like(high_1w, np.nan)
    lower_20 = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        upper_20[i] = np.max(high_1w[i-19:i+1])
        lower_20[i] = np.min(low_1w[i-19:i+1])
    
    # Weekly trend filter (50 EMA)
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    # Align weekly indicators to daily
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(ema50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper + above weekly EMA50 + volume spike
            if close[i] > upper_20_aligned[i] and close[i] > ema50_aligned[i] and vol_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below weekly Donchian lower + below weekly EMA50 + volume spike
            elif close[i] < lower_20_aligned[i] and close[i] < ema50_aligned[i] and vol_spike[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price closes below weekly Donchian lower
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price closes above weekly Donchian upper
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals