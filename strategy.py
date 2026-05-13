#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_TrendFilter
Hypothesis: Donchian(20) breakouts with volume confirmation and EMA(50) trend filter capture strong momentum in both bull and bear markets. 
Volume > 2.0x 20-period average confirms breakout strength. EMA(50) ensures trades align with intermediate-term trend to avoid false breakouts. 
Position size 0.25 balances risk and limits trade frequency (~20-40/year) to minimize fee drag. Works in bull markets via upward breakouts and 
in bear markets via downward breakdowns. Uses 4h timeframe with 1d EMA for trend filter to reduce noise.
"""

name = "4h_Donchian20_Breakout_VolumeSpike_TrendFilter"
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
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # 1d trend filter: EMA(50) on close
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above Donchian high with volume confirmation and uptrend
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian low with volume confirmation and downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian high or trend reverses
            if (close[i] < donchian_high[i]) or \
               (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian low or trend reverses
            if (close[i] > donchian_low[i]) or \
               (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals