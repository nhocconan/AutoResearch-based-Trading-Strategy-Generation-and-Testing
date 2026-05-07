#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Keltner Channel breakout with 1d trend filter and volume confirmation.
# Long when price breaks above upper KC(20,1.5) AND 1d EMA50 rising AND volume > 1.3x 20-period average.
# Short when price breaks below lower KC(20,1.5) AND 1d EMA50 falling AND volume > 1.3x 20-period average.
# Exit when price crosses back inside Keltner Channels.
# Keltner Channels use ATR, which adapts to volatility, making breakouts more meaningful than fixed bands.
# The 1d EMA50 filter ensures alignment with daily trend, reducing whipsaws.
# Volume confirmation ensures institutional participation.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend direction.

name = "6h_KeltnerBreakout_1dEMA50_Volume"
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
    
    # Keltner Channel (20,1.5)
    kc_length = 20
    kc_mult = 1.5
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    atr = pd.Series(tr).rolling(window=kc_length, min_periods=kc_length).mean().values
    ma20 = pd.Series(close).rolling(window=kc_length, min_periods=kc_length).mean().values
    upper_kc = ma20 + (kc_mult * atr)
    lower_kc = ma20 - (kc_mult * atr)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 direction
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kc_length, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ma20[i]) or np.isnan(atr[i]) or np.isnan(upper_kc[i]) or 
            np.isnan(lower_kc[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper KC, 1d EMA50 rising, volume filter
            long_cond = (close[i] > upper_kc[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower KC, 1d EMA50 falling, volume filter
            short_cond = (close[i] < lower_kc[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Keltner Channel (below middle line)
            if close[i] < ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Keltner Channel (above middle line)
            if close[i] > ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals