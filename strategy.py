#!/usr/bin/env python3
# 12H_DONCHIAN_BREAKOUT_1WTREND_VOLUME
# Hypothesis: Donchian channel breakout on 12h with 1-week trend filter and volume confirmation.
# Long when price breaks above 20-period high with price above 1-week EMA and volume > 1.5x average.
# Short when price breaks below 20-period low with price below 1-week EMA and volume > 1.5x average.
# Exit when price returns to opposite Donchian band or trend reverses.
# Designed for 12h timeframe to capture medium-term trends while minimizing trade frequency.

name = "12H_DONCHIAN_BREAKOUT_1WTREND_VOLUME"
timeframe = "12h"
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
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1-week EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    pclose_1w = df_1w['close'].values
    ema1w = pd.Series(pclose_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema1w_aligned = align_htf_to_ltf(prices, df_1w, ema1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # LONG: break above upper band with volume and uptrend
            if close[i] > highest_high[i] and vol_confirm and close[i] > ema1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below lower band with volume and downtrend
            elif close[i] < lowest_low[i] and vol_confirm and close[i] < ema1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to lower band or trend reverses
            if close[i] < lowest_low[i] or close[i] < ema1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to upper band or trend reverses
            if close[i] > highest_high[i] or close[i] > ema1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals