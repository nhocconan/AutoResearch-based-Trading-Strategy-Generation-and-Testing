#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 12h EMA50 rising AND volume > 1.5x 20 EMA
# Short when price breaks below Donchian(20) low AND 12h EMA50 falling AND volume > 1.5x 20 EMA
# Uses 4h timeframe for optimal trade frequency, Donchian for structure, 12h EMA for trend,
# volume confirmation to avoid false breakouts. Designed for 19-50 trades/year with discrete sizing (0.25).
# Works in bull markets via longs in strong uptrends and bear markets via shorts in strong downtrends.

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_rising = np.diff(ema_50_12h, prepend=ema_50_12h[0]) > 0  # Rising trend
    ema_50_12h_falling = np.diff(ema_50_12h, prepend=ema_50_12h[0]) < 0  # Falling trend
    
    # Align 12h EMA50 trend to 4h timeframe
    ema_50_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_rising.astype(float))
    ema_50_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_falling.astype(float))
    
    # Calculate 4h Donchian(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume confirmation (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_rising_aligned[i]) or np.isnan(ema_50_12h_falling_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Price breaks above Donchian high AND 12h EMA50 rising AND volume spike
            if (close[i] > donchian_high[i] and 
                ema_50_12h_rising_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Price breaks below Donchian low AND 12h EMA50 falling AND volume spike
            elif (close[i] < donchian_low[i] and 
                  ema_50_12h_falling_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian low OR 12h EMA50 falls
            if (close[i] < donchian_low[i] or 
                ema_50_12h_falling_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian high OR 12h EMA50 rises
            if (close[i] > donchian_high[i] or 
                ema_50_12h_rising_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals