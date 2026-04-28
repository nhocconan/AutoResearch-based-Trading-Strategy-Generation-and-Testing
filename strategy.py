#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout for direction, 1h for entry timing.
# Enter long when price breaks above 4h 20-period high with volume > 1.5x 20-bar average.
# Enter short when price breaks below 4h 20-period low with volume > 1.5x 20-bar average.
# Exit on opposite Donchian breakout or when price crosses 4h EMA50.
# Uses 4h for signal direction (reduces trade frequency) and 1h for precise entry/exit.
# Volume confirmation filters weak breakouts. Discrete position sizing (0.20) limits drawdown.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
# Works in bull (strong breakouts) and bear (strong breakdowns) via symmetric long/short logic.

name = "1h_Donchian20_Breakout_4hDirection_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian and EMA50
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h EMA50 for exit filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: >1.5x 20-bar average volume (1h)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > 4h 20-period high, volume confirm
            if price > highest_20_aligned[i] and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: price < 4h 20-period low, volume confirm
            elif price < lowest_20_aligned[i] and vol_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit
            # Exit on 4h Donchian breakdown or price crosses below 4h EMA50
            if price < lowest_20_aligned[i] or price < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - hold or exit
            # Exit on 4h Donchian breakout or price crosses above 4h EMA50
            if price > highest_20_aligned[i] or price > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals