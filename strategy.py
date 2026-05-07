#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1w EMA20 AND volume > 1.5x 20-day average.
# Short when price breaks below Donchian(20) low AND price < 1w EMA20 AND volume > 1.5x 20-day average.
# Uses 1w EMA for trend filter to avoid counter-trend trades and volume for momentum confirmation.
# Designed for low trade frequency (target: 10-20 trades/year) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via trend-filtered shorts.
name = "1d_Donchian20_1wEMA20_Volume"
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
    
    # Load 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Load 1d data for Donchian channels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current volume > 1.5x 20-day EMA
    vol_ema_20_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio_1d = np.where(vol_ema_20_1d > 0, df_1d['volume'].values / vol_ema_20_1d, 1.0)
    vol_condition = vol_ratio_1d > 1.5
    vol_condition_aligned = align_htf_to_ltf(prices, df_1d, vol_condition)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_condition_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above Donchian high, above 1w EMA20, volume confirmation
            long_condition = (close[i] > donchian_high_aligned[i]) and \
                           (close[i] > ema_20_1w_aligned[i]) and \
                           vol_condition_aligned[i]
            # Short condition: break below Donchian low, below 1w EMA20, volume confirmation
            short_condition = (close[i] < donchian_low_aligned[i]) and \
                            (close[i] < ema_20_1w_aligned[i]) and \
                            vol_condition_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or falls below 1w EMA20
            if (close[i] < donchian_low_aligned[i]) or (close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or rises above 1w EMA20
            if (close[i] > donchian_high_aligned[i]) or (close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals