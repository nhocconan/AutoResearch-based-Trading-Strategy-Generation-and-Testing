#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Donchian levels: Upper = 20-period high, Lower = 20-period low (using prior 1d candle)
- Long: Close > Upper + volume > 1.5x 20-period avg + price > 1w EMA50
- Short: Close < Lower + volume > 1.5x 20-period avg + price < 1w EMA50
- Exit: Opposite breakout (Close < Upper for long, Close > Lower for short) or EMA50 trend flip
- Uses Donchian for structure, volume for conviction, 1w EMA50 for HTF trend filter
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- EMA50 provides stronger trend filter than EMA34 to reduce false breakouts in choppy markets
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
    
    # Volume confirmation: > 1.5x 20-period average (balanced to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate prior 1d Donchian channels (20-period)
    # Need prior 20 days of OHLC for each 1d bar
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (using prior day's close for look-ahead safety)
    high_20_aligned = align_htf_to_ltf(prices, prices, high_20)
    low_20_aligned = align_htf_to_ltf(prices, prices, low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close > Upper + volume confirmation + price > 1w EMA50
            if (close[i] > high_20_aligned[i] and 
                volume_confirm and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Lower + volume confirmation + price < 1w EMA50
            elif (close[i] < low_20_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Upper OR price < 1w EMA50 (trend flip)
            if close[i] < high_20_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > Lower OR price > 1w EMA50 (trend flip)
            if close[i] > low_20_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0