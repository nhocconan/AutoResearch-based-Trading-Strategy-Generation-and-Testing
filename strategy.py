#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly trend filter (EMA20) and volume confirmation.
# Long when: Close > Upper Donchian (20-day high) AND EMA20(1w) rising AND volume > 1.5 * EMA20(volume).
# Short when: Close < Lower Donchian (20-day low) AND EMA20(1w) falling AND volume > 1.5 * EMA20(volume).
# Exit when price crosses back below/above the 10-day EMA.
# Designed for low trade frequency (target: 10-25/year) to minimize fee drift and improve generalization.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.
name = "1d_Donchian_1wEMA20_Volume"
timeframe = "1d"
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
    
    # Load weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA20 on weekly close
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Rising if current > previous, falling if current < previous
    ema_20_rising = np.zeros_like(ema_20_1w, dtype=bool)
    ema_20_falling = np.zeros_like(ema_20_1w, dtype=bool)
    ema_20_rising[1:] = ema_20_1w[1:] > ema_20_1w[:-1]
    ema_20_falling[1:] = ema_20_1w[1:] < ema_20_1w[:-1]
    
    ema_20_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_20_rising)
    ema_20_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_20_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or np.isnan(ema_10[i]) or 
            np.isnan(ema_20_rising_aligned[i]) or np.isnan(ema_20_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Upper Donchian AND EMA20(1w) rising AND volume spike
            long_condition = (close[i] > upper_donchian[i]) and ema_20_rising_aligned[i] and volume_spike[i]
            # Short: Close < Lower Donchian AND EMA20(1w) falling AND volume spike
            short_condition = (close[i] < lower_donchian[i]) and ema_20_falling_aligned[i] and volume_spike[i]
            
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