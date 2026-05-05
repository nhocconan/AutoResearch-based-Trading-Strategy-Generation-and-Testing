#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w EMA34 trend filter + volume spike confirmation
# Long when: price breaks above 12h Donchian(20) high AND 1w EMA34 is rising (trend up) AND volume > 2x 20-period MA
# Short when: price breaks below 12h Donchian(20) low AND 1w EMA34 is falling (trend down) AND volume > 2x 20-period MA
# Exit when: price returns to 12h Donchian(20) midpoint OR opposite breakout occurs
# Uses Donchian for structure, weekly EMA for trend bias, volume for conviction
# Timeframe: 12h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1wEMA34_VolumeConfirm"
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
    
    # Calculate volume confirmation on 12h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) on 12h
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
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need enough data for EMA34
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close
    ema_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Trend up: current EMA > previous EMA
    ema_trend_up = np.zeros(len(ema_34), dtype=bool)
    ema_trend_up[1:] = ema_34[1:] > ema_34[:-1]
    ema_trend_down = np.zeros(len(ema_34), dtype=bool)
    ema_trend_down[1:] = ema_34[1:] < ema_34[:-1]
    
    # Align 1w EMA trend to 12h timeframe
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_up.astype(float))
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_trend_up_aligned[i]) or np.isnan(ema_trend_down_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + EMA trending up + volume filter
            if (donchian_breakout_up[i] and 
                ema_trend_up_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + EMA trending down + volume filter
            elif (donchian_breakout_down[i] and 
                  ema_trend_down_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR short breakout occurs
            if (donchian_revert_mid[i] or donchian_breakout_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR long breakout occurs
            if (donchian_revert_mid[i] or donchian_breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals