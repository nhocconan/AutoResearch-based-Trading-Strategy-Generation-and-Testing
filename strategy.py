#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Keltner Channel breakout with weekly trend filter and volume confirmation.
# Uses 1d EMA(20) as center, ATR(10) for bands, and 1w EMA(50) for trend direction.
# Long when price breaks above upper KC with 1w uptrend and volume > 1.5x 20-period EMA.
# Short when price breaks below lower KC with 1w downtrend and volume confirmation.
# Exit when price returns to middle KC or trend reverses.
# Designed for low trade frequency (15-25/year) to avoid fee drag. Works in both trending and ranging markets via trend filter.

name = "6h_1dKeltner_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner Channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(20)
    ema_20 = np.zeros_like(close_1d)
    ema_20[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_20[i] = (close_1d[i] * 2/21) + (ema_20[i-1] * 19/21)
    
    # Calculate 1d ATR(10)
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.roll(high_1d, 1) - np.roll(low_1d, 1))
    tr3 = np.abs(np.roll(close_1d, 1) - np.roll(low_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_10 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 9:
            atr_10[i] = np.nan
        else:
            atr_10[i] = np.mean(tr[i-9:i+1])
    
    # Calculate 1d Keltner Channel
    upper_kc = ema_20 + 2 * atr_10
    lower_kc = ema_20 - 2 * atr_10
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_50 = np.zeros_like(close_1w)
    ema_50[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_50[i] = (close_1w[i] * 2/51) + (ema_50[i-1] * 49/51)
    
    # Align indicators to 6h timeframe
    upper_kc_aligned = align_htf_to_ltf(prices, df_1d, upper_kc)
    lower_kc_aligned = align_htf_to_ltf(prices, df_1d, lower_kc)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: 6h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_kc_aligned[i]) or 
            np.isnan(lower_kc_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper KC with 1w uptrend and volume confirmation
            if (close[i] > upper_kc_aligned[i] and 
                close_1w[-1] > ema_50_aligned[i] and  # Current 1w close > 1w EMA50 (uptrend)
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower KC with 1w downtrend and volume confirmation
            elif (close[i] < lower_kc_aligned[i] and 
                  close_1w[-1] < ema_50_aligned[i] and  # Current 1w close < 1w EMA50 (downtrend)
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle KC or 1w trend turns down
            if (close[i] <= ema_20_aligned[i] or 
                close_1w[-1] <= ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle KC or 1w trend turns up
            if (close[i] >= ema_20_aligned[i] or 
                close_1w[-1] >= ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals