#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Long when: price > Donchian(20) high AND 12h EMA(50) rising AND volume > 1.5x average volume.
# Short when: price < Donchian(20) low AND 12h EMA(50) falling AND volume > 1.5x average volume.
# Exit when price crosses back through Donchian midpoint.
# Designed for 4h timeframe with tight entry conditions to avoid overtrading.
# Uses 12h for trend direction and volume confirmation to filter false breakouts.
# Works in bull markets via breakouts above Donchian high in uptrend, in bear markets via breakouts below Donchian low in downtrend.
# Volume filter ensures breakouts have conviction, reducing false signals.
name = "4h_DonchianBreakout_12hTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_rising[1:] = ema_50_12h[1:] > ema_50_12h[:-1]
    ema_50_falling[1:] = ema_50_12h[1:] < ema_50_12h[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling)
    
    # Volume confirmation: volume > 1.5x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian to be valid
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Donchian high AND 12h EMA50 rising AND volume confirmation
            long_condition = (close[i] > donchian_high[i]) and ema_50_rising_aligned[i] and volume_confirm[i]
            # Short: price < Donchian low AND 12h EMA50 falling AND volume confirmation
            short_condition = (close[i] < donchian_low[i]) and ema_50_falling_aligned[i] and volume_confirm[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price < Donchian midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price > Donchian midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals