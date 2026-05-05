#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Donchian(20) high AND price > EMA50(12h) AND volume > 2.0x 20-period average
# Short when price breaks below Donchian(20) low AND price < EMA50(12h) AND volume > 2.0x 20-period average
# Exit when price crosses EMA50(12h) in opposite direction OR Donchian(20) middle line
# Donchian channel provides clear price structure, 12h EMA50 filters counter-trend whipsaws, volume spike confirms institutional participation
# Target: 19-50 trades/year per symbol (75-200 total over 4 years) for 4h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels
    if len(high) >= 20 and len(low) >= 20:
        # Rolling max/min for Donchian channels
        high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_high = high_roll
        donchian_low = low_roll
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian(20) high AND price > EMA50(12h) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian(20) low AND price < EMA50(12h) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA50(12h) OR price < Donchian middle (trend flip or mean reversion)
            if (close[i] < ema_50_12h_aligned[i] or 
                close[i] < donchian_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA50(12h) OR price > Donchian middle (trend flip or mean reversion)
            if (close[i] > ema_50_12h_aligned[i] or 
                close[i] > donchian_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals