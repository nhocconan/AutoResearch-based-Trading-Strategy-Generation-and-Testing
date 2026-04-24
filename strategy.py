#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Donchian upper band (20-period) AND close > 12h EMA50 AND volume > 2.0 * median volume
- Short when price breaks below Donchian lower band (20-period) AND close < 12h EMA50 AND volume > 2.0 * median volume
- Exit when price touches Donchian middle band (10-period EMA of high/low) or trend reverses (close crosses 12h EMA50)
- Uses 4h primary timeframe with 12h HTF for trend alignment to reduce whipsaws
- Volume spike filter (2.0x median) ensures breakout legitimacy, reducing fakeouts
- Donchian channels provide objective breakout levels with proven edge in BTC/ETH
- Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band, trend up (close > EMA50), volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, trend down (close < EMA50), volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches Donchian middle band OR trend reversal (close < EMA50)
            if close[i] <= donchian_mid[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches Donchian middle band OR trend reversal (close > EMA50)
            if close[i] >= donchian_mid[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0