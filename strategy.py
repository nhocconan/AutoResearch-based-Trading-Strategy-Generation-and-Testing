#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION_12HTREND
# Hypothesis: Donchian channel breakouts capture trend continuation. In 12h uptrend (EMA50), go long on upper band breakout with volume spike; in downtrend, go short on lower band breakout. Volume confirmation filters false breakouts. Trend filter ensures alignment with higher timeframe momentum. Works in both bull and bear markets by avoiding counter-trend trades. Target: 20-40 trades/year on 4h timeframe.

name = "4H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION_12HTREND"
timeframe = "4h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # EMA50 for 12h trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channel (20-period) on 4h
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_avg = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        vol_avg[i] = np.mean(volume[i - lookback + 1:i + 1])
    volume_spike = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 12h uptrend + price breaks above Donchian upper band + volume spike
            if (close[i] > ema50_12h_aligned[i] and 
                close[i] > highest_high[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 12h downtrend + price breaks below Donchian lower band + volume spike
            elif (close[i] < ema50_12h_aligned[i] and 
                  close[i] < lowest_low[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band (stop/reverse)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band (stop/reverse)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals