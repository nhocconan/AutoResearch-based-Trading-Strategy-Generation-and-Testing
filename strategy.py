#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-week trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND 1-week EMA50 > EMA200 AND volume spike.
# Short when: price breaks below Donchian(20) low AND 1-week EMA50 < EMA200 AND volume spike.
# Uses Donchian breakouts for entry, weekly EMA trend filter for direction bias, volume for momentum confirmation.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag and improve generalization.
# Works in bull markets via upward breakouts with bullish weekly trend and in bear markets via downward breakouts with bearish weekly trend.
name = "12h_Donchian20_WeeklyTrend_Volume"
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
    
    # Donchian channel (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter (EMA50 and EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for weekly EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + bullish weekly trend + volume spike
            long_condition = (close[i] > donchian_high[i]) and (ema_50_1w_aligned[i] > ema_200_1w_aligned[i]) and volume_spike[i]
            # Short: breakdown below Donchian low + bearish weekly trend + volume spike
            short_condition = (close[i] < donchian_low[i]) and (ema_50_1w_aligned[i] < ema_200_1w_aligned[i]) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or weekly trend turns bearish
            if (close[i] < donchian_low[i]) or (ema_50_1w_aligned[i] < ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or weekly trend turns bullish
            if (close[i] > donchian_high[i]) or (ema_50_1w_aligned[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals