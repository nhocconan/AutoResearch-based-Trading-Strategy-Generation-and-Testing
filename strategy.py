#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(50) trend + volume confirmation
# Long when price breaks above upper Donchian(20) AND 1w EMA(50) rising AND volume > 1.5x avg volume
# Short when price breaks below lower Donchian(20) AND 1w EMA(50) falling AND volume > 1.5x avg volume
# Exit when price crosses back through the opposite Donchian band
# Designed for 1d timeframe with low trade frequency (target: 20-50/year) to avoid fee drag.
# Uses 1w for trend direction and volume confirmation to avoid false breakouts.
# Works in bull markets via breakouts in uptrend, in bear markets via breakdowns in downtrend.
name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    upper_dc = np.zeros(n)
    lower_dc = np.zeros(n)
    for i in range(20, n):
        upper_dc[i] = np.max(high[i-20:i])
        lower_dc[i] = np.min(low[i-20:i])
    
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
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Donchian + 1w EMA50 rising + volume confirmation
            long_condition = (close[i] > upper_dc[i]) and ema_50_rising_aligned[i] and volume_confirm[i]
            # Short: break below lower Donchian + 1w EMA50 falling + volume confirmation
            short_condition = (close[i] < lower_dc[i]) and ema_50_falling_aligned[i] and volume_confirm[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses back below lower Donchian
            if close[i] < lower_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses back above upper Donchian
            if close[i] > upper_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals