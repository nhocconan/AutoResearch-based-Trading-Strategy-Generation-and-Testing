#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Long when: price breaks above Donchian(20) upper AND 12h EMA(50) rising AND volume > 20-period average
# Short when: price breaks below Donchian(20) lower AND 12h EMA(50) falling AND volume > 20-period average
# Exit when price returns to Donchian midpoint
# Designed for 4h timeframe with moderate trade frequency (target: 20-50/year) to avoid fee drag
# Uses 12h for trend direction and volume for confirmation to avoid false breakouts
# Works in bull markets via upward breakouts in uptrend, in bear markets via downward breakouts in downtrend
# Volume filter avoids low-liquidity breakouts that often fail
name = "4h_Donchian_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper band + 12h EMA50 rising + volume confirmation
            long_breakout = close[i] > highest_high[i]
            long_volume = volume[i] > vol_avg[i]
            long_condition = long_breakout and ema_50_rising_aligned[i] and long_volume
            
            # Short: break below lower band + 12h EMA50 falling + volume confirmation
            short_breakout = close[i] < lowest_low[i]
            short_volume = volume[i] > vol_avg[i]
            short_condition = short_breakout and ema_50_falling_aligned[i] and short_volume
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: return to midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: return to midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals