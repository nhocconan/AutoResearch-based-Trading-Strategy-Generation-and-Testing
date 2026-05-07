# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with 1-week trend filter (EMA34) and volume confirmation.
# Long when: Close > Upper Donchian (10-period high) AND EMA34(1w) rising AND volume > 1.5 * EMA20(volume).
# Short when: Close < Lower Donchian (10-period low) AND EMA34(1w) falling AND volume > 1.5 * EMA20(volume).
# Exit when price crosses back below/above the 5-period EMA.
# Designed for low trade frequency (target: 10-25/year) to minimize fee drift and improve generalization.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.
name = "1d_Donchian_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel: 10-period high/low
    upper_donchian = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lower_donchian = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # EMA5 for exit
    ema_5 = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Load 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA34 on 1w close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Rising if current > previous, falling if current < previous
    ema_34_rising = np.zeros_like(ema_34_1w, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1w, dtype=bool)
    ema_34_rising[1:] = ema_34_1w[1:] > ema_34_1w[:-1]
    ema_34_falling[1:] = ema_34_1w[1:] < ema_34_1w[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_34_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or np.isnan(ema_5[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Upper Donchian AND EMA34(1w) rising AND volume spike
            long_condition = (close[i] > upper_donchian[i]) and ema_34_rising_aligned[i] and volume_spike[i]
            # Short: Close < Lower Donchian AND EMA34(1w) falling AND volume spike
            short_condition = (close[i] < lower_donchian[i]) and ema_34_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < EMA5
            if close[i] < ema_5[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > EMA5
            if close[i] > ema_5[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals