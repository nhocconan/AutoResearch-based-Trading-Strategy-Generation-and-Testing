#!/usr/bin/env python3
"""
12h_TRIX_ZeroCross_VolumeSpike_1dTrend
Hypothesis: TRIX (12-period) crossing above zero line with volume > 1.5x 20-period average in uptrend (price > EMA34 1d) signals long momentum; crossing below zero with volume confirmation in downtrend (price < EMA34 1d) signals short momentum. Uses 12h timeframe to limit trade frequency (target 50-150 total trades over 4 years) and avoid fee drag. Volume and trend filters ensure trades align with momentum and reduce whipsaw in sideways markets.
"""

name = "12h_TRIX_ZeroCross_VolumeSpike_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate TRIX (12-period triple EMA) on close prices
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix_raw = np.full_like(close, np.nan)
    trix_raw[12:] = 100 * (ema3[12:] - ema3[11:-1]) / ema3[11:-1]
    
    # Get 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # cooldown counter to prevent immediate re-entry
    
    for i in range(20, n):  # Start after TRIX warmup
        # Decrease cooldown if active
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            # LONG: TRIX crosses above zero with volume confirmation in uptrend
            if not np.isnan(trix_raw[i-1]) and not np.isnan(trix_raw[i]) and \
               trix_raw[i-1] <= 0 and trix_raw[i] > 0 and \
               volume_confirmed[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume confirmation in downtrend
            elif not np.isnan(trix_raw[i-1]) and not np.isnan(trix_raw[i]) and \
                 trix_raw[i-1] >= 0 and trix_raw[i] < 0 and \
                 volume_confirmed[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses back below zero or trend weakens
            if not np.isnan(trix_raw[i-1]) and not np.isnan(trix_raw[i]) and \
               trix_raw[i-1] > 0 and trix_raw[i] <= 0:
                signals[i] = 0.0
                position = 0
                cooldown = 2  # 2-bar cooldown after exit
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses back above zero or trend weakens
            if not np.isnan(trix_raw[i-1]) and not np.isnan(trix_raw[i]) and \
                 trix_raw[i-1] < 0 and trix_raw[i] >= 0:
                signals[i] = 0.0
                position = 0
                cooldown = 2  # 2-bar cooldown after exit
            else:
                signals[i] = -0.25
    
    return signals