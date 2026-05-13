#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation (>1.5x 20-bar avg volume), and chop regime filter (CHOP(14) < 38.2 = trending). Uses discrete position sizing (0.25) to minimize fee churn. Designed for BTC/ETH robustness in both bull and bear markets via confluence of price structure, trend, volume, and regime filters. Targets 75-200 total trades over 4 years on 4h timeframe.

name = "4h_Donchian20_Breakout_12hEMA50_VolumeChopRegime_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Choppiness Index (CHOP) on 14-period for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h_arr, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h_arr, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop_12h = 100 * np.log10(atr_sum / true_range_sum) / np.log10(14)
    chop_12h = np.where(true_range_sum == 0, 50, chop_12h)  # avoid div by zero
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate Donchian channels (20-period) for breakout signals
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(chop_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high, price > 12h EMA50, volume spike (>1.5x avg), trending regime (CHOP < 38.2)
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                chop_12h_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low, price < 12h EMA50, volume spike (>1.5x avg), trending regime (CHOP < 38.2)
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  chop_12h_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price drops below Donchian low (reversal) OR chop becomes too high (choppy market)
            if (close[i] < donchian_low[i] or 
                chop_12h_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price rises above Donchian high (reversal) OR chop becomes too high (choppy market)
            if (close[i] > donchian_high[i] or 
                chop_12h_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals