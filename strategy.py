#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter, volume confirmation (>1.8x 20-bar avg volume), and weekly choppiness regime filter (CHOP > 61.8 = range -> avoid entries, CHOP < 38.2 = trend -> allow breakout entries). Uses 1d timeframe to target 30-100 total trades over 4 years. Weekly trend and regime filters reduce false signals in bear markets. Discrete position sizing (0.25) minimizes fee churn.

name = "1d_Donchian20_Breakout_1wEMA50_VolumeChopRegime_v1"
timeframe = "1d"
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly Choppiness Index (CHOP) on 14-period for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w_arr, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w_arr, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop_1w = 100 * np.log10(atr_sum / true_range_sum) / np.log10(14)
    chop_1w = np.where(true_range_sum == 0, 50, chop_1w)  # avoid div by zero
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate Donchian channels (20-period) from daily OHLC
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(chop_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above 20-day high, price > 1w EMA50, volume spike (>1.8x avg), trending regime (CHOP < 38.2)
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i] and 
                chop_1w_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below 20-day low, price < 1w EMA50, volume spike (>1.8x avg), trending regime (CHOP < 38.2)
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i] and 
                  chop_1w_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price drops below 20-day low (reversal) OR chop becomes too high (choppy market)
            if (close[i] < lowest_low[i] or 
                chop_1w_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price rises above 20-day high (reversal) OR chop becomes too high (choppy market)
            if (close[i] > highest_high[i] or 
                chop_1w_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals