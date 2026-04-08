#!/usr/bin/env python3
# 1h_4h1d_ema_crossover_volume_v2
# Hypothesis: EMA crossover on 1h with 4h/1d trend filter and volume confirmation.
# Uses 4h EMA crossover for primary direction and 1d EMA for regime filter.
# Enters on 1h EMA cross only when higher timeframes agree and volume confirms.
# Targets 15-30 trades/year by requiring multiple timeframe alignment + volume surge.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_ema_crossover_volume_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA crossover for primary direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA for regime filter (bull/bear)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: 1h volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: 4h EMA bearish cross OR stoploss hit
            if ema20_4h_aligned[i] < ema50_4h_aligned[i] or close[i] < close[i-1] - 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: 4h EMA bullish cross OR stoploss hit
            if ema20_4h_aligned[i] > ema50_4h_aligned[i] or close[i] > close[i-1] + 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: 4h bullish cross + 1d uptrend (price > EMA200) + volume surge
            if (ema20_4h_aligned[i] > ema50_4h_aligned[i] and  # 4h bullish cross
                close[i] > ema200_1d_aligned[i] and  # 1d uptrend regime
                vol_surge):
                position = 1
                signals[i] = 0.20
            # Short entry: 4h bearish cross + 1d downtrend (price < EMA200) + volume surge
            elif (ema20_4h_aligned[i] < ema50_4h_aligned[i] and  # 4h bearish cross
                  close[i] < ema200_1d_aligned[i] and  # 1d downtrend regime
                  vol_surge):
                position = -1
                signals[i] = -0.20
    
    return signals