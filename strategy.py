#!/usr/bin/env python3
# 4h_1d_keltner_breakout_volume_v1
# Strategy: 4h Keltner Channel breakout with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Keltner Channel breakouts capture momentum in trending markets.
# Long when price breaks above upper Keltner (EMA + ATR*2) and 1d EMA50 up, short when breaks below lower Keltner (EMA - ATR*2) and 1d EMA50 down.
# Volume spike (1.5x 20-period average) confirms momentum. Designed for 20-40 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0  # no previous close for first bar
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # EMA(20) for Keltner middle
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channels
    kc_upper = ema_20 + (2 * atr)
    kc_lower = ema_20 - (2 * atr)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: 1d EMA50 direction (using slope)
        if i >= 61:
            ema_now = ema_50_1d_aligned[i]
            ema_prev = ema_50_1d_aligned[i-1]
            uptrend = ema_now > ema_prev
            downtrend = ema_now < ema_prev
        else:
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout signals
        breakout_up = close[i] > kc_upper[i]
        breakout_down = close[i] < kc_lower[i]
        
        # Entry logic: Keltner breakout + volume spike + 1d trend alignment
        if (breakout_up and volume_spike[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (breakout_down and volume_spike[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout or trend change
        elif position == 1 and (breakout_down or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_up or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals