#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1w trend filter and 1d volume confirmation.
# Long when: Williams %R < -80 AND 1w EMA(50) rising AND 1d volume > 1.5x 20-period average
# Short when: Williams %R > -20 AND 1w EMA(50) falling AND 1d volume > 1.5x 20-period average
# Exit when Williams %R crosses back to -50.
# Designed for 12h timeframe with low trade frequency (target: 15-30/year) to avoid fee drag.
# Uses 1w for trend direction and 1d for volume confirmation to avoid false signals.
# Williams %R identifies overbought/oversold conditions, effective in both trending and ranging markets.
# Volume filter ensures participation, reducing false breakouts.
# Trend filter ensures trading in direction of higher timeframe momentum.
name = "12h_WilliamsR_1wEMA50_1dVolume"
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
    
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_rising[1:] = ema_50_1w[1:] > ema_50_1w[:-1]
    ema_50_falling[1:] = ema_50_1w[1:] < ema_50_1w[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling)
    
    # 1d volume > 1.5x 20-period average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (1.5 * vol_ma_20)
    
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 AND 1w EMA50 rising AND volume confirmation
            long_condition = (williams_r[i] < -80) and ema_50_rising_aligned[i] and volume_confirm_aligned[i]
            # Short: Williams %R > -20 AND 1w EMA50 falling AND volume confirmation
            short_condition = (williams_r[i] > -20) and ema_50_falling_aligned[i] and volume_confirm_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R > -50
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R < -50
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals