#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Choppiness_Regime_Donchian_Breakout_1wTrend"
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
    
    # Calculate 1w trend: EMA21
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate ATR(14) for position sizing and stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14-period)
    # Chop = 100 * log10(sum(ATR(1)) / (n * log10(highest_high - lowest_low)))
    atr1 = tr  # ATR(1) is true range
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = highest_high14 - lowest_low14
    # Avoid division by zero
    range14 = np.where(range14 == 0, 1e-10, range14)
    chop = 100 * np.log10(sum_atr1 / (14 * np.log10(range14)))
    chop = np.where(np.isnan(chop), 50, chop)  # fill NaN with neutral value
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(high_max20[i]) or 
            np.isnan(low_min20[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout + uptrend (price > 1w EMA21) + chop > 61.8 (range)
            long_cond = (close[i] > high_max20[i]) and \
                        (close[i] > ema_21_1w_aligned[i]) and \
                        (chop[i] > 61.8)
            # Short: Donchian breakdown + downtrend (price < 1w EMA21) + chop > 61.8 (range)
            short_cond = (close[i] < low_min20[i]) and \
                         (close[i] < ema_21_1w_aligned[i]) and \
                         (chop[i] > 61.8)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low OR chop < 38.2 (trending)
            if close[i] < low_min20[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Donchian high OR chop < 38.2 (trending)
            if close[i] > high_max20[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals