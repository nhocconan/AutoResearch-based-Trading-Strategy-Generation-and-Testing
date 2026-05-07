#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with daily trend filter (EMA34) and volume confirmation.
# Long when: Close > Weekly Upper Donchian (20-period high) AND EMA34(1d) rising AND volume > 1.5 * EMA20(volume).
# Short when: Close < Weekly Lower Donchian (20-period low) AND EMA34(1d) falling AND volume > 1.5 * EMA20(volume).
# Exit when price crosses back below/above the daily 10-period EMA.
# Target: 10-25 trades/year per symbol to minimize fee drag and improve generalization.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.
name = "1d_WeeklyDonchian_1dEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Exit: EMA10 on close
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Weekly Donchian Channel: 20-period high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_donchian_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_donchian_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    upper_donchian_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_donchian_1w)
    lower_donchian_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_donchian_1w)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(upper_donchian_1w_aligned[i]) or np.isnan(lower_donchian_1w_aligned[i]) or 
            np.isnan(ema_10[i]) or np.isnan(ema_34_rising) or np.isnan(ema_34_falling) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Weekly Upper Donchian AND EMA34(1d) rising AND volume spike
            long_condition = (close[i] > upper_donchian_1w_aligned[i]) and ema_34_rising[i] and volume_spike[i]
            # Short: Close < Weekly Lower Donchian AND EMA34(1d) falling AND volume spike
            short_condition = (close[i] < lower_donchian_1w_aligned[i]) and ema_34_falling[i] and volume_spike[i]
            
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