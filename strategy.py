#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 12h EMA50 rising AND volume > 1.5x 20 EMA
# Short when price breaks below Donchian(20) low AND 12h EMA50 falling AND volume > 1.5x 20 EMA
# Uses discrete sizing (0.25) to minimize fee drag. Designed for 20-50 trades/year on 4h.
# Works in bull markets via longs on upward breakouts and bear markets via shorts on downward breakouts.
# Volume confirmation avoids false breakouts in low-participation markets.
# 12h EMA50 ensures we only trade in the direction of the intermediate trend.

name = "4h_Donchian20_12hEMA50_Trend_Volume"
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
    ema_50_12h_rising = np.diff(ema_50_12h, prepend=0) > 0  # Rising if current > previous
    
    # Align 12h EMA50 trend to 4h timeframe
    ema_50_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_rising.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume EMA(20) for volume spike filter
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_rising_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND 12h EMA50 rising AND volume spike
            if (close[i] > donchian_high[i] and 
                ema_50_12h_rising_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND 12h EMA50 falling AND volume spike
            elif (close[i] < donchian_low[i] and 
                  ema_50_12h_rising_aligned[i] < 0.5 and  # Falling if not rising
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR 12h EMA50 turns falling
            if (close[i] < donchian_low[i] or 
                ema_50_12h_rising_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR 12h EMA50 turns rising
            if (close[i] > donchian_high[i] or 
                ema_50_12h_rising_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals