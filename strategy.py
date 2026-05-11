#!/usr/bin/env python3
name = "12h_1D_Camarilla_R1_S1_Breakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla R1, S1 from previous day's OHLC
    # Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Calculate R1 and S1 for each day
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1-day EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    daily_uptrend = close > ema_34_1d_aligned
    
    # Volume spike filter (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # ensure EMA34 and volume MA ready
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(daily_uptrend[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1, daily uptrend, volume spike
            if close[i] > r1_aligned[i] and daily_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, daily downtrend, volume spike
            elif close[i] < s1_aligned[i] and not daily_uptrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 or trend reverses
            if close[i] < s1_aligned[i] or not daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 or trend reverses
            if close[i] > r1_aligned[i] or daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals