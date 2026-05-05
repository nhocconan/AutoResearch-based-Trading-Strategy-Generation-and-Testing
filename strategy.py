#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation
# Long when: price breaks above 1d Donchian(20) high AND 1w EMA34 is rising AND volume > 2x 20-period MA
# Short when: price breaks below 1d Donchian(20) low AND 1w EMA34 is falling AND volume > 2x 20-period MA
# Exit when: price returns to 1d Donchian(20) midpoint
# Uses Donchian for structure, weekly EMA for bias, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_1wEMA34_VolumeConfirm"
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
    
    # Calculate volume confirmation on 1d using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) on 1d
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (highest_high + lowest_low) / 2.0
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Donchian breakout signals
    donchian_breakout_up = (close > highest_high) & (np.roll(close, 1) <= np.roll(highest_high, 1))
    donchian_breakout_down = (close < lowest_low) & (np.roll(close, 1) >= np.roll(lowest_low, 1))
    donchian_revert_mid = np.abs(close - donchian_mid) < 0.001 * close  # approximate midpoint return
    
    # Get 1w data ONCE before loop for EMA34 calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # EMA34 trend: rising if current > previous, falling if current < previous
    ema_34_rising = np.zeros(len(ema_34_1w), dtype=bool)
    ema_34_falling = np.zeros(len(ema_34_1w), dtype=bool)
    ema_34_rising[1:] = ema_34_1w[1:] > ema_34_1w[:-1]
    ema_34_falling[1:] = ema_34_1w[1:] < ema_34_1w[:-1]
    
    # Align 1w EMA34 trend to 1d timeframe
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_34_rising.astype(float))
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_34_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + EMA34 rising + volume filter
            if (donchian_breakout_up[i] and 
                ema_34_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + EMA34 falling + volume filter
            elif (donchian_breakout_down[i] and 
                  ema_34_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint
            if donchian_revert_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint
            if donchian_revert_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals