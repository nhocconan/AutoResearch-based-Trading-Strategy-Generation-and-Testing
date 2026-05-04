#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter (EMA34) and volume confirmation
# Long when price breaks above 20-period Donchian high AND 1d EMA34 is rising AND volume > 1.5x 20-period volume EMA
# Short when price breaks below 20-period Donchian low AND 1d EMA34 is falling AND volume > 1.5x 20-period volume EMA
# Uses 1d EMA34 for major trend filter to reduce whipsaw vs shorter HTF, targeting 12-37 trades/year on 12h.
# Volume spike filter (1.5x) is strict to avoid overtrading. Donchian breakouts provide clear structure.
# Works in bull markets via longs in bullish 1d EMA34 regime and bear markets via shorts in bearish 1d EMA34 regime.

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_1d_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_1d_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_1d_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    # Align 1d EMA34 trend to 12h timeframe
    ema_34_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_rising.astype(float))
    ema_34_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_falling.astype(float))
    
    # Calculate 20-period Donchian channels
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(ema_34_1d_rising_aligned[i]) or np.isnan(ema_34_1d_falling_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND 1d EMA34 rising AND volume spike
            if (close[i] > donchian_h[i] and 
                ema_34_1d_rising_aligned[i] > 0.5 and  # 1d EMA34 rising
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND 1d EMA34 falling AND volume spike
            elif (close[i] < donchian_l[i] and 
                  ema_34_1d_falling_aligned[i] > 0.5 and  # 1d EMA34 falling
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low OR 1d EMA34 turns falling
            if (close[i] < donchian_l[i] or 
                ema_34_1d_falling_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR 1d EMA34 turns rising
            if (close[i] > donchian_h[i] or 
                ema_34_1d_rising_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals