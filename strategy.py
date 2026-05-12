#!/usr/bin/env python3
name = "4h_TRIX_13_VolumeSpike_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D TREND: EMA34 ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === TRIX(13) ON CLOSE ===
    ema1 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema2 = pd.Series(ema1).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema3 = pd.Series(ema2).ewm(span=13, adjust=False, min_periods=13).mean().values
    trix_raw = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0
    trix = trix_raw * 100  # percentage
    
    # === VOLUME SPIKE (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(trix[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX > 0 (bullish momentum) + ABOVE 1D EMA34 + VOLUME SPIKE
            if (trix[i] > 0 and 
                close[i] > ema34_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX < 0 (bearish momentum) + BELOW 1D EMA34 + VOLUME SPIKE
            elif (trix[i] < 0 and 
                  close[i] < ema34_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: TRIX turns negative OR price below 1D EMA34
            if trix[i] < 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns positive OR price above 1D EMA34
            if trix[i] > 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals