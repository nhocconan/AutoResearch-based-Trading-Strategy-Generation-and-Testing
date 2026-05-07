#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND 1d EMA(50) rising AND volume > 1.5x 20-period average
# Short when: price breaks below Donchian(20) low AND 1d EMA(50) falling AND volume > 1.5x 20-period average
# Exit when price crosses back through Donchian(20) midline or opposite breakout occurs.
# Designed for 4h timeframe with moderate trade frequency (target: 20-50/year) to avoid fee drag.
# Uses 1d for trend direction to avoid counter-trend trades and volume to confirm breakout strength.
# Works in bull markets via breakouts in uptrend, in bear markets via breakdowns in downtrend.
# Volume filter ensures breakouts have conviction and reduces false signals.
name = "4h_DonchianBreakout_1dTrend_VolumeConfirm"
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
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Donchian midline for exit
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume average (20-period)
    vol_avg = np.full_like(volume, np.nan)
    for i in range(lookback - 1, n):
        vol_avg[i] = np.mean(volume[i - lookback + 1:i + 1])
    vol_threshold = 1.5  # volume must be 1.5x average
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND 1d EMA50 rising AND volume confirmation
            long_condition = (close[i] > highest_high[i]) and ema_50_rising_aligned[i] and (volume[i] > vol_avg[i] * vol_threshold)
            # Short: price breaks below Donchian low AND 1d EMA50 falling AND volume confirmation
            short_condition = (close[i] < lowest_low[i]) and ema_50_falling_aligned[i] and (volume[i] > vol_avg[i] * vol_threshold)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below Donchian midline OR opposite breakout with volume
            exit_condition = (close[i] < donchian_mid[i]) or \
                           (close[i] < lowest_low[i] and volume[i] > vol_avg[i] * vol_threshold)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above Donchian midline OR opposite breakout with volume
            exit_condition = (close[i] > donchian_mid[i]) or \
                           (close[i] > highest_high[i] and volume[i] > vol_avg[i] * vol_threshold)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals