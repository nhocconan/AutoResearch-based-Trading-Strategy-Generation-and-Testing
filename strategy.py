#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 1d EMA34 up AND volume > 1.5x 20 EMA
# Short when price breaks below Donchian(20) low AND 1d EMA34 down AND volume > 1.5x 20 EMA
# Exit on opposite Donchian break or trend change
# Designed for 12-37 trades/year with discrete sizing (0.25).
# Works in bull markets via longs on breakouts and bear markets via shorts on breakdowns.

name = "12h_Donchian20_1dEMA34_VolumeConfirm"
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
    
    # Get 1d data for HTF EMA34 filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_up = ema_34_1d > np.roll(ema_34_1d, 1)
    ema_34_1d_up[0] = False
    ema_34_1d_down = ema_34_1d < np.roll(ema_34_1d, 1)
    ema_34_1d_down[0] = False
    
    # Align 1d EMA34 trend to 12h timeframe
    ema_34_1d_up_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_up.astype(float))
    ema_34_1d_down_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_down.astype(float))
    
    # Calculate 12h Donchian(20)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_up_aligned[i]) or np.isnan(ema_34_1d_down_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high AND 1d EMA34 up AND volume spike
            if (close[i] > highest_20[i] and 
                ema_34_1d_up_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low AND 1d EMA34 down AND volume spike
            elif (close[i] < lowest_20[i] and 
                  ema_34_1d_down_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below Donchian low OR 1d EMA34 turns down
            if (close[i] < lowest_20[i] or 
                ema_34_1d_down_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above Donchian high OR 1d EMA34 turns up
            if (close[i] > highest_20[i] or 
                ema_34_1d_up_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals