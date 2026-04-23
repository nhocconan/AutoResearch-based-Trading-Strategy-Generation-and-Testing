#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Donchian levels: Upper = 20-period high, Lower = 20-period low (using prior 1d candle)
- Long: Close > Upper + volume > 2.0x 20-period avg + price > 1w EMA50
- Short: Close < Lower + volume > 2.0x 20-period avg + price < 1w EMA50
- Exit: Opposite breakout (Close < Upper for long, Close > Lower for short) or EMA50 trend flip
- Uses Donchian for structure, volume for conviction, 1w EMA50 for HTF trend filter
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by EMA50)
"""

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
    
    # Volume confirmation: > 2.0x 20-period average (tighter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate prior 1d Donchian(20) levels
    # Need prior 1d OHLC for each 1d bar
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian(20) Upper = rolling max(high, 20), Lower = rolling min(low, 20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper_1d = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (using prior 1d close for look-ahead safety)
    donchian_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(donchian_upper_1d_aligned[i]) or
            np.isnan(donchian_lower_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > Upper + volume confirmation + price > 1w EMA50
            if (close[i] > donchian_upper_1d_aligned[i] and 
                volume_confirm and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Lower + volume confirmation + price < 1w EMA50
            elif (close[i] < donchian_lower_1d_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Upper OR price < 1w EMA50 (trend flip)
            if close[i] < donchian_upper_1d_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > Lower OR price > 1w EMA50 (trend flip)
            if close[i] > donchian_lower_1d_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0