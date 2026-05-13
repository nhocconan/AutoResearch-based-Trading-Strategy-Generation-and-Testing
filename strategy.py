#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 12h Donchian upper channel AND price > 1d EMA50 AND volume > 1.5x 20-period average volume.
# Short when price breaks below 12h Donchian lower channel AND price < 1d EMA50 AND volume > 1.5x 20-period average volume.
# Exit when price crosses the 12h Donchian middle line (20-period average of high/low).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness in trending markets by capturing breakouts with institutional volume while avoiding false signals in low-volume or ranging conditions.

name = "12h_DonchianBreakout_EMA50_VolumeConfirm_v1"
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
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian upper: highest high over 20 periods
    donch_hi = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low over 20 periods
    donch_lo = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Donchian middle: average of upper and lower
    donch_mid = (donch_hi + donch_lo) / 2.0
    
    # Align HTF Donchian levels to LTF
    donch_hi_aligned = align_htf_to_ltf(prices, df_12h, donch_hi)
    donch_lo_aligned = align_htf_to_ltf(prices, df_12h, donch_lo)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(donch_hi_aligned[i]) or 
            np.isnan(donch_lo_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # LONG: price breaks above Donchian upper AND price > 1d EMA50 AND volume confirmation
            if (close[i] > donch_hi_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower AND price < 1d EMA50 AND volume confirmation
            elif (close[i] < donch_lo_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below Donchian middle
            if close[i] < donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above Donchian middle
            if close[i] > donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals