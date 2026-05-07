#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla R1/S1 breakout with 1-day trend filter (EMA34) and volume spike.
# Long when: Close > Camarilla R1 AND EMA34(1d) rising AND volume > 2.0 * EMA20(volume).
# Short when: Close < Camarilla S1 AND EMA34(1d) falling AND volume > 2.0 * EMA20(volume).
# Exit when price crosses back below/above Camarilla S1/R1.
# Uses daily trend filter to align with higher timeframe momentum, reducing whipsaw.
# Volume spike ensures momentum confirmation. Target: 20-40 trades/year per symbol.
name = "4h_Camarilla_R1S1_1dEMA34_VolumeSpike"
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
    
    # Camarilla levels from previous day (using previous day's high, low, close)
    # Calculate daily OHLC from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Extract daily OHLC arrays
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = np.zeros_like(daily_close)
    camarilla_s1 = np.zeros_like(daily_close)
    
    for i in range(len(daily_close)):
        if i == 0:
            camarilla_r1[i] = np.nan
            camarilla_s1[i] = np.nan
        else:
            # Previous day's OHLC
            ph = daily_high[i-1]
            pl = daily_low[i-1]
            pc = daily_close[i-1]
            range_ = ph - pl
            camarilla_r1[i] = pc + (range_ * 1.1 / 12)
            camarilla_s1[i] = pc - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (using previous day's values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Camarilla R1 AND EMA34(1d) rising AND volume spike
            long_condition = (close[i] > camarilla_r1_aligned[i]) and ema_34_rising_aligned[i] and volume_spike[i]
            # Short: Close < Camarilla S1 AND EMA34(1d) falling AND volume spike
            short_condition = (close[i] < camarilla_s1_aligned[i]) and ema_34_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < Camarilla S1
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > Camarilla R1
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals