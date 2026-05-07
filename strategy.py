#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w trend filter and 1d volume confirmation.
# Long when: price > Donchian(20) high AND 1w EMA(34) rising AND 1d volume > 1.5x SMA(10) volume
# Short when: price < Donchian(20) low AND 1w EMA(34) falling AND 1d volume > 1.5x SMA(10) volume
# Exit when price crosses back to Donchian(20) midline.
# Designed for 12h timeframe with low trade frequency (target: 12-37/year) to avoid fee drag.
# Uses 1w for trend direction and 1d for volume confirmation to avoid choppy markets.
# Works in bull markets via breakouts in uptrend, in bear markets via breakdowns in downtrend.
# Volume filter ensures only significant breakouts are traded.
name = "12h_Donchian20_1wEMA34_1dVolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (highest_20 + lowest_20) / 2.0
    
    # 1w EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1w, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1w, dtype=bool)
    ema_34_rising[1:] = ema_34_1w[1:] > ema_34_1w[:-1]
    ema_34_falling[1:] = ema_34_1w[1:] < ema_34_1w[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_34_falling)
    
    # 1d volume confirmation: volume > 1.5x SMA(10) volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_sma_10 = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_threshold = vol_sma_10 * 1.5
    
    vol_1d = pd.Series(volume_1d)
    vol_confirm = (vol_1d > vol_threshold).values
    
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for Donchian(20)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(mid_20[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Donchian high AND 1w EMA34 rising AND volume confirmation
            long_condition = (close[i] > highest_20[i]) and ema_34_rising_aligned[i] and vol_confirm_aligned[i]
            # Short: price < Donchian low AND 1w EMA34 falling AND volume confirmation
            short_condition = (close[i] < lowest_20[i]) and ema_34_falling_aligned[i] and vol_confirm_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price < Donchian midline
            if close[i] < mid_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price > Donchian midline
            if close[i] > mid_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals