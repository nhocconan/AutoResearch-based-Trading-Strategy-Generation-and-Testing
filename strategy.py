#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price > upper Donchian channel AND close > 12h EMA50 AND volume > 1.5 * avg volume.
# Short when price < lower Donchian channel AND close < 12h EMA50 AND volume > 1.5 * avg volume.
# Exit when price crosses opposite Donchian band OR volume < avg volume.
# Uses discrete position sizing (0.25) to limit fee churn.
# Designed for ~20-50 trades/year by requiring confluence of breakout, trend, and volume spike.
# Works in both bull and bear markets by capturing strong directional moves with trend and volatility filters.

name = "4h_Donchian20_Breakout_12hEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper Donchian AND close > 12h EMA50 AND volume spike
            if close[i] > highest_high[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < lower Donchian AND close < 12h EMA50 AND volume spike
            elif close[i] < lowest_low[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < lower Donchian OR volume drops below average
            if close[i] < lowest_low[i] or volume[i] < avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > upper Donchian OR volume drops below average
            if close[i] > highest_high[i] or volume[i] < avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals