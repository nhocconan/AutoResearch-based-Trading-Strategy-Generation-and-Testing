#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Filtered"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # need 34 for EMA + 1 for prev day
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close for Camarilla calculation
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels from previous day
    hl_range = high_1d - low_1d
    camarilla_r1 = prev_close_1d + hl_range * 1.083
    camarilla_s1 = prev_close_1d - hl_range * 1.083
    
    # Align Camarilla levels (previous day's levels available at 4h bar open)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d trend filter: EMA 34
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: 20-period average (conservative)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    # Choppiness filter: avoid choppy markets (CHOP > 61.8)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - low)
    tr3 = np.abs(np.roll(low, 1) - high)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_sum = pd.Series(max_high - min_low).rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(atr * atr_period / range_sum) / np.log10(atr_period)
    chop_filter = chop < 61.8  # trending market
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20, atr_period)
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 + above 1d EMA + volume + trending
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                vol_filter[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + below 1d EMA + volume + trending
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  vol_filter[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below S1 or below 1d EMA or choppy market
            if (close[i] < camarilla_s1_aligned[i] or 
                close[i] < ema_1d_aligned[i] or 
                not chop_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above R1 or above 1d EMA or choppy market
            if (close[i] > camarilla_r1_aligned[i] or 
                close[i] > ema_1d_aligned[i] or 
                not chop_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals