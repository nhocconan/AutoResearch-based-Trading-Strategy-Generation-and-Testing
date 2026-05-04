#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 1w EMA50 rising AND volume > 1.5x 20 EMA
# Short when price breaks below Donchian(20) low AND 1w EMA50 falling AND volume > 1.5x 20 EMA
# Uses 1d timeframe for lower frequency, Donchian for structure, 1w EMA50 for major trend filter,
# volume confirmation to avoid false signals. Designed for 7-25 trades/year with discrete sizing (0.25).
# Works in bull markets via longs in strong uptrends and bear markets via shorts in strong downtrends.

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.diff(ema_50_1w, prepend=ema_50_1w[0]) > 0  # Rising when positive
    ema_50_falling = np.diff(ema_50_1w, prepend=ema_50_1w[0]) < 0  # Falling when negative
    
    # Align 1w EMA50 trend to 1d timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising.astype(float))
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling.astype(float))
    
    # Calculate 1d Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND EMA50 rising AND volume spike
            if (close[i] > donchian_high[i] and 
                ema_50_rising_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND EMA50 falling AND volume spike
            elif (close[i] < donchian_low[i] and 
                  ema_50_falling_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR EMA50 falls
            if (close[i] < donchian_low[i] or 
                ema_50_rising_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR EMA50 rises
            if (close[i] > donchian_high[i] or 
                ema_50_falling_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals