#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND 1d EMA(50) rising AND volume > 1.5x 20-period average
# Short when: price breaks below Donchian(20) low AND 1d EMA(50) falling AND volume > 1.5x 20-period average
# Exit when price crosses back through Donchian(20) midline.
# Uses Donchian for clear breakout structure, 1d EMA for trend filter to avoid counter-trend trades,
# volume confirmation to ensure breakout strength, and midline exit for controlled exits.
# Designed for 4h timeframe with moderate trade frequency (target: 25-40/year) to balance opportunity and cost.
name = "4h_DonchianBreakout_1dEMA50_VolumeConfirm"
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
    
    # Donchian Channel (20-period)
    lookback = 20
    dc_high = np.full(n, np.nan)
    dc_low = np.full(n, np.nan)
    dc_mid = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        dc_high[i] = np.max(high[i - lookback + 1:i + 1])
        dc_low[i] = np.min(low[i - lookback + 1:i + 1])
        dc_mid[i] = (dc_high[i] + dc_low[i]) / 2
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_1d, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1d, dtype=bool)
    ema_50_rising[1:] = ema_50_1d[1:] > ema_50_1d[:-1]
    ema_50_falling[1:] = ema_50_1d[1:] < ema_50_1d[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback - 1)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(dc_mid[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high AND 1d EMA50 rising AND volume confirmation
            long_condition = (close[i] > dc_high[i]) and ema_50_rising_aligned[i] and volume_confirm[i]
            # Short: break below Donchian low AND 1d EMA50 falling AND volume confirmation
            short_condition = (close[i] < dc_low[i]) and ema_50_falling_aligned[i] and volume_confirm[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian midline
            if close[i] < dc_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian midline
            if close[i] > dc_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals