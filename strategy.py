#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_1dEMA34
Hypothesis: 4-hour Donchian channel breakouts with volume confirmation and daily trend alignment (EMA34) capture sustained momentum. Exits on opposite Donchian breach or trend reversal. Uses discrete position sizing (0.25) to limit trades to ~20-50/year, reducing fee drag. Works in bull markets via breakouts with trend and in bear via mean-reversion at channel extremes when trend weakens.
"""

name = "4h_Donchian_Breakout_Volume_Trend_1dEMA34"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above upper Donchian with volume and uptrend
            if (close[i] > upper[i] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below lower Donchian with volume and downtrend
            elif (close[i] < lower[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian or trend turns down
            if (close[i] < lower[i]) or \
               (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian or trend turns up
            if (close[i] > upper[i]) or \
               (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals