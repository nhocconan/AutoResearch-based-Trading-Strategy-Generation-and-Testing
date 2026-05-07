#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 12-hour trend filter (EMA50) and volume confirmation.
# Long when: Close > Upper Donchian (20-period high) AND EMA50(12h) rising AND volume > 1.5 * EMA20(volume).
# Short when: Close < Lower Donchian (20-period low) AND EMA50(12h) falling AND volume > 1.5 * EMA20(volume).
# Exit when price crosses back below/above the 10-period EMA.
# Designed for low trade frequency (target: 25-40/year) to minimize fee drift and improve generalization.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.
name = "4h_Donchian_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel: 20-period high/low
    upper_donchian = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA10 for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA50 on 12h close
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_rising[1:] = ema_50_12h[1:] > ema_50_12h[:-1]
    ema_50_falling[1:] = ema_50_12h[1:] < ema_50_12h[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or np.isnan(ema_10[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Upper Donchian AND EMA50(12h) rising AND volume spike
            long_condition = (close[i] > upper_donchian[i]) and ema_50_rising_aligned[i] and volume_spike[i]
            # Short: Close < Lower Donchian AND EMA50(12h) falling AND volume spike
            short_condition = (close[i] < lower_donchian[i]) and ema_50_falling_aligned[i] and volume_spike[i]
            
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