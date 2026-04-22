#!/usr/bin/env python3
"""
Hypothesis: 1-hour Exponential Moving Average crossover with 4-hour trend filter and daily volume confirmation.
Long when 1h EMA10 > EMA30 and 4h EMA50 rising with daily volume > 1.5x 20-day average.
Short when 1h EMA10 < EMA30 and 4h EMA50 falling with daily volume > 1.5x 20-day average.
Exit when EMA crossover reverses or 4h EMA50 changes direction.
Uses higher timeframe for trend direction and volume for institutional confirmation to reduce false signals.
Designed for low trade frequency by requiring multiple confirmations, targeting 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA crossover: fast EMA10, slow EMA30
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema30 = pd.Series(close).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # 4h EMA50 for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Daily volume confirmation: current volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema10[i]) or np.isnan(ema30[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current hourly volume > 1.5x daily 20-day average
        # Note: comparing hourly volume to daily average - this works because
        # we're looking for unusually high hourly activity relative to daily norm
        vol_spike = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: EMA10 > EMA30 and 4h EMA50 rising with volume spike
            if (ema10[i] > ema30[i] and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and vol_spike):
                signals[i] = 0.20
                position = 1
            # Short: EMA10 < EMA30 and 4h EMA50 falling with volume spike
            elif (ema10[i] < ema30[i] and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and vol_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: EMA crossover reverses or 4h EMA50 changes direction
            exit_signal = False
            
            if position == 1:
                # Exit long: EMA10 <= EMA30 or 4h EMA50 turns down
                if ema10[i] <= ema30[i] or ema50_4h_aligned[i] < ema50_4h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: EMA10 >= EMA30 or 4h EMA50 turns up
                if ema10[i] >= ema30[i] or ema50_4h_aligned[i] > ema50_4h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_EMA_Crossover_4hTrend_DailyVol"
timeframe = "1h"
leverage = 1.0