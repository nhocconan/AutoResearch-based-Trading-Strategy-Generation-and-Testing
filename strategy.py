#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND 12h EMA50 is rising AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian(20) low AND 12h EMA50 is falling AND volume > 1.5x 20-bar avg
# Exit when price retouches Donchian(20) midpoint OR volume drops below average
# Target: 20-50 trades/year via tight Donchian breakout + trend + volume confluence
# Works in bull markets via long breakouts, bear markets via short breakouts

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_prev = np.roll(ema_50_12h, 1)
    ema_50_12h_prev[0] = np.nan
    ema_50_12h_rising = ema_50_12h > ema_50_12h_prev
    ema_50_12h_falling = ema_50_12h < ema_50_12h_prev
    
    # Align 12h EMA trend to 4h timeframe
    ema_50_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_rising)
    ema_50_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_falling)
    
    # Calculate Donchian(20) channels on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian(20) to be valid
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_12h_rising_aligned[i]) or np.isnan(ema_50_12h_falling_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_rising = ema_50_12h_rising_aligned[i]
        ema_falling = ema_50_12h_falling_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND 12h EMA50 rising AND volume confirmation
            if close[i] > donchian_high[i] and ema_rising and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND 12h EMA50 falling AND volume confirmation
            elif close[i] < donchian_low[i] and ema_falling and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches Donchian midpoint OR volume drops
            if close[i] <= donchian_mid[i] or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches Donchian midpoint OR volume drops
            if close[i] >= donchian_mid[i] or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals