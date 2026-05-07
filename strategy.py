#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian upper band AND 1d EMA50 trend up AND volume spike.
# Short when price breaks below 20-period Donchian lower band AND 1d EMA50 trend down AND volume spike.
# Uses Donchian for breakout signals, 1d EMA for trend filter, volume for momentum confirmation.
# Designed for low trade frequency (target: 15-25/year) to minimize fee drain and improve generalization.
# Works in bull markets via upward breakouts in uptrend and in bear markets via downward breakouts in downtrend.
name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > upper band AND 1d EMA up AND volume spike
            long_condition = (close[i] > high_20[i]) and (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and volume_spike[i]
            # Short: price < lower band AND 1d EMA down AND volume spike
            short_condition = (close[i] < low_20[i]) and (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price < lower band OR 1d EMA turns down
            if (close[i] < low_20[i]) or (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price > upper band OR 1d EMA turns up
            if (close[i] > high_20[i]) or (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals