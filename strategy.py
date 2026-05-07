#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian channel breakout with 1-week EMA trend filter and volume confirmation.
# Long when: Close breaks above Donchian upper (20) AND EMA50(1w) rising AND volume > 1.5 * EMA20(volume).
# Short when: Close breaks below Donchian lower (20) AND EMA50(1w) falling AND volume > 1.5 * EMA20(volume).
# Exit when price crosses back below/above the 10-day EMA.
# Designed for low trade frequency (target: 7-25/year) to minimize fee drag and improve generalization.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.
name = "1d_Donchian_20_1wEMA50_VolumeBreakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Exit EMA (10-period)
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: EMA20 of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ema_20
    
    # Load 1-week data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA50 on 1w close
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_rising[1:] = ema_50_1w[1:] > ema_50_1w[:-1]
    ema_50_falling[1:] = ema_50_1w[1:] < ema_50_1w[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema_10[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Donchian upper AND EMA50(1w) rising AND volume spike
            long_condition = (close[i] > high_max[i]) and ema_50_rising_aligned[i] and (volume[i] > volume_threshold[i])
            # Short: Close < Donchian lower AND EMA50(1w) falling AND volume spike
            short_condition = (close[i] < low_min[i]) and ema_50_falling_aligned[i] and (volume[i] > volume_threshold[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < EMA10
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > EMA10
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals