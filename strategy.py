# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12-hour exponential moving average crossover with volume confirmation
# - Long when 9-period EMA crosses above 21-period EMA on 12h timeframe, volume > 1.5x 48-period average
# - Short when 9-period EMA crosses below 21-period EMA on 12h timeframe, volume > 1.5x 48-period average
# - Uses EMA crossover on higher timeframe to reduce whipsaws and capture sustained trends
# - Volume confirmation ensures breakouts have institutional participation
# - Position size 0.25 to balance risk and returns
# - Target: 50-100 trades over 4 years (12-25/year) to minimize fee drag
# - Works in both bull and bear markets by following established trends with confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 9 and 21 period EMA on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema9_12h = close_12h_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21_12h = close_12h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate EMA crossover signals on 12h timeframe
    ema_crossover = np.zeros(len(df_12h))
    ema_crossover[9:] = np.where(ema9_12h[9:] > ema21_12h[9:], 1, 
                                np.where(ema9_12h[9:] < ema21_12h[9:], -1, 0))
    
    # Align EMA crossover to 4h timeframe (waits for 12h bar to close)
    ema_crossover_aligned = align_htf_to_ltf(prices, df_12h, ema_crossover)
    
    # Volume filter: 48-period average (2 days of 4h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(ema_crossover_aligned[i]) or np.isnan(vol_ma[i]):
            continue
        
        if position == 0:
            # Long: EMA bullish crossover on 12h with volume confirmation
            if (ema_crossover_aligned[i] == 1 and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: EMA bearish crossover on 12h with volume confirmation
            elif (ema_crossover_aligned[i] == -1 and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: EMA bearish crossover on 12h (trend change)
            if ema_crossover_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: EMA bullish crossover on 12h (trend change)
            if ema_crossover_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12h_EMA9_21_Crossover_Volume"
timeframe = "4h"
leverage = 1.0