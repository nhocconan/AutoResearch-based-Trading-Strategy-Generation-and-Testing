#!/usr/bin/env python3
name = "12h_Wilson_Trend_Reversal"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def wilson_price_oscillator(high, low, close, period):
    """Wilson Price Oscillator: normalized momentum oscillator"""
    wpo = np.zeros_like(close, dtype=float)
    wpo[:] = np.nan
    
    for i in range(period, len(close)):
        # Calculate smoothed high and low
        high_max = np.max(high[i-period+1:i+1])
        low_min = np.min(low[i-period+1:i+1])
        
        if high_max != low_min:
            wpo[i] = (close[i] - low_min) / (high_max - low_min) * 100 - 50
        else:
            wpo[i] = 0
    
    return wpo

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA for trend filter (1w HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily Wilson Price Oscillator for momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    wpo_14 = wilson_price_oscillator(high_1d, low_1d, close_1d, 14)
    wpo_14_aligned = align_htf_to_ltf(prices, df_1d, wpo_14)
    
    # Volume filter: current volume > 1.8 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(wpo_14_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition
        vol_condition = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # LONG: WPO crosses above -20 with weekly uptrend and volume
            if (wpo_14_aligned[i] > -20 and 
                wpo_14_aligned[i-1] <= -20 and
                close[i] > ema21_1w_aligned[i] and 
                vol_condition):
                signals[i] = 0.25
                position = 1
            # SHORT: WPO crosses below 20 with weekly downtrend and volume
            elif (wpo_14_aligned[i] < 20 and 
                  wpo_14_aligned[i-1] >= 20 and
                  close[i] < ema21_1w_aligned[i] and 
                  vol_condition):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: WPO crosses below 20 or weekly trend reversal
            if (wpo_14_aligned[i] < 20 or 
                close[i] < ema21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: WPO crosses above -20 or weekly trend reversal
            if (wpo_14_aligned[i] > -20 or 
                close[i] > ema21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals