#!/usr/bin/env python3
# Hypothesis: 12h Donchian channel breakout with 1d EMA50 trend filter, volume confirmation (>1.8x 20-bar avg volume), and choppiness regime filter (CHOP < 38.2 = trend -> allow breakout entries). Uses 12h timeframe to target 50-150 total trades over 4 years. Donchian breakouts capture strong momentum, EMA50 filter ensures trend alignment, volume spike confirms conviction, and chop filter avoids range-bound false signals. Discrete position sizing (0.25) minimizes fee churn. Works in both bull (follows upward breakouts) and bear (captures downward breakdowns) markets.

name = "12h_Donchian20_Breakout_1dEMA50_VolumeChopRegime_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian Channel (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate Choppiness Index (CHOP) on 14-period for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / true_range_sum) / np.log10(14)
    chop = np.where(true_range_sum == 0, 50, chop)  # avoid div by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close > Donchian High (breakout), close > 1d EMA50, volume spike (>1.8x avg), trending regime (CHOP < 38.2)
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i] and 
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Close < Donchian Low (breakdown), close < 1d EMA50, volume spike (>1.8x avg), trending regime (CHOP < 38.2)
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i] and 
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if close < Donchian Low (breakdown) OR chop becomes too high (choppy market)
            if (close[i] < donchian_low[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if close > Donchian High (breakout) OR chop becomes too high (choppy market)
            if (close[i] > donchian_high[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals