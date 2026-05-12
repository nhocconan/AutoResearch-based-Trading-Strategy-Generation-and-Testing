#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_ChopFilter_W1Trend
Hypothesis: TRIX (12-period) combined with volume spikes and Choppiness regime filter on 12h timeframe, with 1w trend filter, captures momentum bursts in both bull and bear markets while avoiding whipsaws. Uses discrete position sizing (0.25) to limit turnover. Target: 20-40 trades/year per symbol.
"""

name = "12h_TRIX_VolumeSpike_ChopFilter_W1Trend"
timeframe = "12h"
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
    
    # Volume spike: >1.8x 30-period average (selective)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # TRIX (12-period) on close
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # Choppiness index (14-period) for regime filter
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr14.sum() / (highest_high - lowest_low)) / np.log10(14)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    
    # Weekly trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Align TRIX and Chop to 12h (they're already calculated on 12h data)
    # TRIX and Chop are calculated on LTF, so no alignment needed
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(trix[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above 0 + volume spike + chop < 61.8 (trending) + price > weekly EMA34
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                volume_spike[i] and 
                chop[i] < 61.8 and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below 0 + volume spike + chop < 61.8 (trending) + price < weekly EMA34
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  volume_spike[i] and 
                  chop[i] < 61.8 and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below 0 OR chop > 61.8 (choppy) OR weekly trend breaks
            if (trix[i] < 0 and trix[i-1] >= 0) or \
               (chop[i] > 61.8) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above 0 OR chop > 61.8 (choppy) OR weekly trend breaks
            if (trix[i] > 0 and trix[i-1] <= 0) or \
               (chop[i] > 61.8) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals