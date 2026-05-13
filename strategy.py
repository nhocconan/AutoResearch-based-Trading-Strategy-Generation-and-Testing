#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_1wTrend
Hypothesis: TRIX (12) on 12h timeframe captures momentum shifts, confirmed by volume spikes and 1-week trend alignment.
Works in bull markets by catching strong momentum and in bear markets by avoiding false signals via trend filter.
Targets 15-25 trades/year to minimize fee drag.
"""

name = "12h_TRIX_VolumeSpike_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX calculation on 12h data
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA1: 12-period EMA of close
    ema1 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2: 12-period EMA of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3: 12-period EMA of EMA2
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX: 1-period percent change of EMA3
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1]
    
    # Align TRIX to 12h chart
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    
    # 1-week trend: EMA50 on 1w data
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.5x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: TRIX crosses above zero with volume confirmation and uptrend
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                volume_filter[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume confirmation and downtrend
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  volume_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend reverses
            if trix_aligned[i] < 0 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend reverses
            if trix_aligned[i] > 0 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals