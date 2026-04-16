#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 4h EMA50 > EMA100 AND volume > 1.5x 4h avg volume
# Short when price breaks below Donchian(20) low AND 4h EMA50 < EMA100 AND volume > 1.5x 4h avg volume
# Exit when price crosses back below/above Donchian midline
# Uses 4h for trend direction and volume confirmation, 1h only for entry/exit timing
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h EMA50 and EMA100 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_4h = pd.Series(close_4h).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_100_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_100_4h)
    
    # === 4h Volume Average for confirmation ===
    volume_4h = df_4h['volume'].values
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values  # 20 periods of 4h = ~3.3 days
    vol_avg_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    
    # === 1h Donchian Channel (20-period) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_100_4h_aligned[i]) or
            np.isnan(vol_avg_4h_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_50 = ema_50_4h_aligned[i]
        ema_100 = ema_100_4h_aligned[i]
        vol_confirm = volume[i] > vol_avg_4h_aligned[i] * 1.5  # 1.5x 4h average volume
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price crosses below Donchian midline
            if price < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            # Exit when price crosses above Donchian midline
            if price > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian high AND 4h EMA50 > EMA100 AND volume confirmation
            if price > highest_20[i] and ema_50 > ema_100 and vol_confirm:
                signals[i] = 0.20
                position = 1
                continue
            # Short when: price breaks below Donchian low AND 4h EMA50 < EMA100 AND volume confirmation
            elif price < lowest_20[i] and ema_50 < ema_100 and vol_confirm:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_4hEMA50_100_Vol1.5x"
timeframe = "1h"
leverage = 1.0