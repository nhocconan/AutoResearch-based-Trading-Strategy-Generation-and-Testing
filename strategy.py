#!/usr/bin/env python3
"""
1h_Trend_Filtered_Breakout
Hypothesis: 1h breakouts at 4h Donchian channels filtered by 1d EMA50 trend and volume spike.
Trades only in direction of daily trend to avoid counter-trend whipsaws in bear markets.
Volume confirmation reduces false breakouts. Session filter (08-20 UTC) avoids low-liquidity hours.
Designed for 15-30 trades/year to minimize fee drag while capturing trend momentum.
"""

name = "1h_Trend_Filtered_Breakout"
timeframe = "1h"
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
    
    # Pre-compute hour filter for efficiency
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # === 4h Data for Donchian Breakout Levels ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 20-period Donchian channels on 4h
    high_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h (wait for 4h bar close)
    donchian_high = align_htf_to_ltf(prices, df_4h, high_max)
    donchian_low = align_htf_to_ltf(prices, df_4h, low_min)
    
    # === 1d Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 50-period EMA for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Filter: 1.5x 24-period EMA on 1h ===
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_spike = volume > vol_ema24 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1d EMA50)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with uptrend and volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below Donchian low with downtrend and volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low (reversal signal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: close above Donchian high (reversal signal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals